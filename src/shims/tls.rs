//! Implement thread-local storage.

use std::collections::btree_map::Entry as BTreeEntry;
use std::collections::hash_map::Entry as HashMapEntry;
use std::collections::BTreeMap;

use log::trace;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty;
use rustc_target::abi::{HasDataLayout, Size};
use rustc_target::spec::abi::Abi;

use crate::*;

pub type TlsKey = u128;

#[derive(Clone, Debug)]
pub struct TlsEntry<'tcx> {
    /// The data for this key. None is used to represent NULL.
    /// (We normalize this early to avoid having to do a NULL-ptr-test each time we access the data.)
    data: BTreeMap<ThreadId, Scalar<Tag>>,
    dtor: Option<ty::Instance<'tcx>>,
}

#[derive(Clone, Debug)]
struct RunningDtorsState {
    /// The last TlsKey used to retrieve a TLS destructor. `None` means that we
    /// have not tried to retrieve a TLS destructor yet or that we already tried
    /// all keys.
    last_dtor_key: Option<TlsKey>,
}

#[derive(Debug)]
pub struct TlsData<'tcx> {
    /// The Key to use for the next thread-local allocation.
    next_key: TlsKey,

    /// pthreads-style thread-local storage.
    keys: BTreeMap<TlsKey, TlsEntry<'tcx>>,

    /// A single per thread destructor of the thread local storage (that's how
    /// things work on macOS) with a data argument.
    macos_thread_dtors: BTreeMap<ThreadId, (ty::Instance<'tcx>, Scalar<Tag>)>,

    /// State for currently running TLS dtors. If this map contains a key for a
    /// specific thread, it means that we are in the "destruct" phase, during
    /// which some operations are UB.
    dtors_running: FxHashMap<ThreadId, RunningDtorsState>,
}

impl<'tcx> Default for TlsData<'tcx> {
    fn default() -> Self {
        TlsData {
            next_key: 1, // start with 1 as we must not use 0 on Windows
            keys: Default::default(),
            macos_thread_dtors: Default::default(),
            dtors_running: Default::default(),
        }
    }
}

impl<'tcx> TlsData<'tcx> {
    /// Generate a new TLS key with the given destructor.
    /// `max_size` determines the integer size the key has to fit in.
    pub fn create_tls_key(
        &mut self,
        dtor: Option<ty::Instance<'tcx>>,
        max_size: Size,
    ) -> InterpResult<'tcx, TlsKey> {
        let new_key = self.next_key;
        self.next_key += 1;
        self.keys.try_insert(new_key, TlsEntry { data: Default::default(), dtor }).unwrap();
        trace!("New TLS key allocated: {} with dtor {:?}", new_key, dtor);

        if max_size.bits() < 128 && new_key >= (1u128 << max_size.bits() as u128) {
            throw_unsup_format!("we ran out of TLS key space");
        }
        Ok(new_key)
    }

    pub fn delete_tls_key(&mut self, key: TlsKey) -> InterpResult<'tcx> {
        match self.keys.remove(&key) {
            Some(_) => {
                trace!("TLS key {} removed", key);
                Ok(())
            }
            None => throw_ub_format!("removing a non-existig TLS key: {}", key),
        }
    }

    pub fn load_tls(
        &self,
        key: TlsKey,
        thread_id: ThreadId,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx, Scalar<Tag>> {
        match self.keys.get(&key) {
            Some(TlsEntry { data, .. }) => {
                let value = data.get(&thread_id).copied();
                trace!("TLS key {} for thread {:?} loaded: {:?}", key, thread_id, value);
                Ok(value.unwrap_or_else(|| Scalar::null_ptr(cx).into()))
            }
            None => throw_ub_format!("loading from a non-existing TLS key: {}", key),
        }
    }

    pub fn store_tls(
        &mut self,
        key: TlsKey,
        thread_id: ThreadId,
        new_data: Scalar<Tag>,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx> {
        match self.keys.get_mut(&key) {
            Some(TlsEntry { data, .. }) => {
                if new_data.to_machine_usize(cx)? != 0 {
                    trace!("TLS key {} for thread {:?} stored: {:?}", key, thread_id, new_data);
                    data.insert(thread_id, new_data);
                } else {
                    trace!("TLS key {} for thread {:?} removed", key, thread_id);
                    data.remove(&thread_id);
                }
                Ok(())
            }
            None => throw_ub_format!("storing to a non-existing TLS key: {}", key),
        }
    }

    /// Set the thread wide destructor of the thread local storage for the given
    /// thread. This function is used to implement `_tlv_atexit` shim on MacOS.
    ///
    /// Thread wide dtors are available only on MacOS. There is one destructor
    /// per thread as can be guessed from the following comment in the
    /// [`_tlv_atexit`
    /// implementation](https://github.com/opensource-apple/dyld/blob/195030646877261f0c8c7ad8b001f52d6a26f514/src/threadLocalVariables.c#L389):
    ///
    ///     // NOTE: this does not need locks because it only operates on current thread data
    pub fn set_macos_thread_dtor(
        &mut self,
        thread: ThreadId,
        dtor: ty::Instance<'tcx>,
        data: Scalar<Tag>,
    ) -> InterpResult<'tcx> {
        if self.dtors_running.contains_key(&thread) {
            // UB, according to libstd docs.
            throw_ub_format!(
                "setting thread's local storage destructor while destructors are already running"
            );
        }
        if self.macos_thread_dtors.insert(thread, (dtor, data)).is_some() {
            throw_unsup_format!(
                "setting more than one thread local storage destructor for the same thread is not supported"
            );
        }
        Ok(())
    }

    /// Returns a dtor, its argument and its index, if one is supposed to run.
    /// `key` is the last dtors that was run; we return the *next* one after that.
    ///
    /// An optional destructor function may be associated with each key value.
    /// At thread exit, if a key value has a non-NULL destructor pointer,
    /// and the thread has a non-NULL value associated with that key,
    /// the value of the key is set to NULL, and then the function pointed
    /// to is called with the previously associated value as its sole argument.
    /// The order of destructor calls is unspecified if more than one destructor
    /// exists for a thread when it exits.
    ///
    /// If, after all the destructors have been called for all non-NULL values
    /// with associated destructors, there are still some non-NULL values with
    /// associated destructors, then the process is repeated.
    /// If, after at least {PTHREAD_DESTRUCTOR_ITERATIONS} iterations of destructor
    /// calls for outstanding non-NULL values, there are still some non-NULL values
    /// with associated destructors, implementations may stop calling destructors,
    /// or they may continue calling destructors until no non-NULL values with
    /// associated destructors exist, even though this might result in an infinite loop.
    fn fetch_tls_dtor(
        &mut self,
        key: Option<TlsKey>,
        thread_id: ThreadId,
    ) -> Option<(ty::Instance<'tcx>, Scalar<Tag>, TlsKey)> {
        use std::ops::Bound::*;

        let thread_local = &mut self.keys;
        let start = match key {
            Some(key) => Excluded(key),
            None => Unbounded,
        };
        for (&key, TlsEntry { data, dtor }) in thread_local.range_mut((start, Unbounded)) {
            match data.entry(thread_id) {
                BTreeEntry::Occupied(entry) => {
                    if let Some(dtor) = dtor {
                        // Set TLS data to NULL, and call dtor with old value.
                        let data_scalar = entry.remove();
                        let ret = Some((*dtor, data_scalar, key));
                        return ret;
                    }
                }
                BTreeEntry::Vacant(_) => {}
            }
        }
        None
    }

    /// Set that dtors are running for `thread`. It is guaranteed not to change
    /// the existing values stored in `dtors_running` for this thread. Returns
    /// `true` if dtors for `thread` are already running.
    fn set_dtors_running_for_thread(&mut self, thread: ThreadId) -> bool {
        match self.dtors_running.entry(thread) {
            HashMapEntry::Occupied(_) => true,
            HashMapEntry::Vacant(entry) => {
                // We cannot just do `self.dtors_running.insert` because that
                // would overwrite `last_dtor_key` with `None`.
                entry.insert(RunningDtorsState { last_dtor_key: None });
                false
            }
        }
    }

    /// Delete all TLS entries for the given thread. This function should be
    /// called after all TLS destructors have already finished.
    fn delete_all_thread_tls(&mut self, thread_id: ThreadId) {
        for TlsEntry { data, .. } in self.keys.values_mut() {
            data.remove(&thread_id);
        }
    }
}

impl<'mir, 'tcx: 'mir> EvalContextPrivExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
trait EvalContextPrivExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    /// Schedule TLS destructors for the main thread on Windows. The
    /// implementation assumes that we do not support concurrency on Windows
    /// yet.
    fn schedule_windows_tls_dtors(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let active_thread = this.get_active_thread();
        assert_eq!(this.get_total_thread_count(), 1, "concurrency on Windows is not supported");
        // Windows has a special magic linker section that is run on certain events.
        // Instead of searching for that section and supporting arbitrary hooks in there
        // (that would be basically https://github.com/rust-lang/miri/issues/450),
        // we specifically look up the static in libstd that we know is placed
        // in that section.
        let thread_callback = this.eval_path_scalar(&[
            "std",
            "sys",
            "windows",
            "thread_local_key",
            "p_thread_callback",
        ])?;
        let thread_callback =
            this.memory.get_fn(this.scalar_to_ptr(thread_callback))?.as_instance()?;

        // The signature of this function is `unsafe extern "system" fn(h: c::LPVOID, dwReason: c::DWORD, pv: c::LPVOID)`.
        let reason = this.eval_path_scalar(&["std", "sys", "windows", "c", "DLL_THREAD_DETACH"])?;
        let ret_place = MPlaceTy::dangling(this.machine.layouts.unit).into();
        this.call_function(
            thread_callback,
            Abi::System { unwind: false },
            &[Scalar::null_ptr(this).into(), reason.into(), Scalar::null_ptr(this).into()],
            Some(&ret_place),
            StackPopCleanup::Root { cleanup: true },
        )?;

        this.enable_thread(active_thread);
        Ok(())
    }

    /// Schedule the MacOS thread destructor of the thread local storage to be
    /// executed. Returns `true` if scheduled.
    ///
    /// Note: It is safe to call this function also on other Unixes.
    fn schedule_macos_tls_dtor(&mut self) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        let thread_id = this.get_active_thread();
        if let Some((instance, data)) = this.machine.tls.macos_thread_dtors.remove(&thread_id) {
            trace!("Running macos dtor {:?} on {:?} at {:?}", instance, data, thread_id);

            let ret_place = MPlaceTy::dangling(this.machine.layouts.unit).into();
            this.call_function(
                instance,
                Abi::C { unwind: false },
                &[data.into()],
                Some(&ret_place),
                StackPopCleanup::Root { cleanup: true },
            )?;

            // Enable the thread so that it steps through the destructor which
            // we just scheduled. Since we deleted the destructor, it is
            // guaranteed that we will schedule it again. The `dtors_running`
            // flag will prevent the code from adding the destructor again.
            this.enable_thread(thread_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Schedule a pthread TLS destructor. Returns `true` if found
    /// a destructor to schedule, and `false` otherwise.
    fn schedule_next_pthread_tls_dtor(&mut self) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        let active_thread = this.get_active_thread();

        assert!(this.has_terminated(active_thread), "running TLS dtors for non-terminated thread");
        // Fetch next dtor after `key`.
        let last_key = this.machine.tls.dtors_running[&active_thread].last_dtor_key.clone();
        let dtor = match this.machine.tls.fetch_tls_dtor(last_key, active_thread) {
            dtor @ Some(_) => dtor,
            // We ran each dtor once, start over from the beginning.
            None => this.machine.tls.fetch_tls_dtor(None, active_thread),
        };
        if let Some((instance, ptr, key)) = dtor {
            this.machine.tls.dtors_running.get_mut(&active_thread).unwrap().last_dtor_key =
                Some(key);
            trace!("Running TLS dtor {:?} on {:?} at {:?}", instance, ptr, active_thread);
            assert!(
                !ptr.to_machine_usize(this).unwrap() != 0,
                "data can't be NULL when dtor is called!"
            );

            let ret_place = MPlaceTy::dangling(this.machine.layouts.unit).into();
            this.call_function(
                instance,
                Abi::C { unwind: false },
                &[ptr.into()],
                Some(&ret_place),
                StackPopCleanup::Root { cleanup: true },
            )?;

            this.enable_thread(active_thread);
            return Ok(true);
        }
        this.machine.tls.dtors_running.get_mut(&active_thread).unwrap().last_dtor_key = None;

        Ok(false)
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    /// Schedule an active thread's TLS destructor to run on the active thread.
    /// Note that this function does not run the destructors itself, it just
    /// schedules them one by one each time it is called and reenables the
    /// thread so that it can be executed normally by the main execution loop.
    ///
    /// Note: we consistently run TLS destructors for all threads, including the
    /// main thread. However, it is not clear that we should run the TLS
    /// destructors for the main thread. See issue:
    /// https://github.com/rust-lang/rust/issues/28129.
    fn schedule_next_tls_dtor_for_active_thread(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let active_thread = this.get_active_thread();
        trace!("schedule_next_tls_dtor_for_active_thread on thread {:?}", active_thread);

        if !this.machine.tls.set_dtors_running_for_thread(active_thread) {
            // This is the first time we got asked to schedule a destructor. The
            // Windows schedule destructor function must be called exactly once,
            // this is why it is in this block.
            if this.tcx.sess.target.os == "windows" {
                // On Windows, we signal that the thread quit by starting the
                // relevant function, reenabling the thread, and going back to
                // the scheduler.
                this.schedule_windows_tls_dtors()?;
                return Ok(());
            }
        }
        // The remaining dtors make some progress each time around the scheduler loop,
        // until they return `false` to indicate that they are done.

        // The macOS thread wide destructor runs "before any TLS slots get
        // freed", so do that first.
        if this.schedule_macos_tls_dtor()? {
            // We have scheduled a MacOS dtor to run on the thread. Execute it
            // to completion and come back here. Scheduling a destructor
            // destroys it, so we will not enter this branch again.
            return Ok(());
        }
        if this.schedule_next_pthread_tls_dtor()? {
            // We have scheduled a pthread destructor and removed it from the
            // destructors list. Run it to completion and come back here.
            return Ok(());
        }

        // All dtors done!
        this.machine.tls.delete_all_thread_tls(active_thread);
        this.thread_terminated()?;

        Ok(())
    }
}
