//! Implement thread-local storage.

use std::collections::btree_map::Entry as BTreeEntry;
use std::collections::BTreeMap;
use std::task::Poll;

use log::trace;

use rustc_middle::ty;
use rustc_target::abi::{HasDataLayout, Size};
use rustc_target::spec::abi::Abi;

use crate::*;

pub type TlsKey = u128;

#[derive(Clone, Debug)]
pub struct TlsEntry<'tcx> {
    /// The data for this key. None is used to represent NULL.
    /// (We normalize this early to avoid having to do a NULL-ptr-test each time we access the data.)
    data: BTreeMap<ThreadId, Scalar<Provenance>>,
    dtor: Option<ty::Instance<'tcx>>,
}

#[derive(Default, Debug)]
struct RunningDtorState {
    /// The last TlsKey used to retrieve a TLS destructor. `None` means that we
    /// have not tried to retrieve a TLS destructor yet or that we already tried
    /// all keys.
    last_key: Option<TlsKey>,
}

#[derive(Debug)]
pub struct TlsData<'tcx> {
    /// The Key to use for the next thread-local allocation.
    next_key: TlsKey,

    /// pthreads-style thread-local storage.
    keys: BTreeMap<TlsKey, TlsEntry<'tcx>>,

    /// A single per thread destructor of the thread local storage (that's how
    /// things work on macOS) with a data argument.
    macos_thread_dtors: BTreeMap<ThreadId, (ty::Instance<'tcx>, Scalar<Provenance>)>,
}

impl<'tcx> Default for TlsData<'tcx> {
    fn default() -> Self {
        TlsData {
            next_key: 1, // start with 1 as we must not use 0 on Windows
            keys: Default::default(),
            macos_thread_dtors: Default::default(),
        }
    }
}

impl<'tcx> TlsData<'tcx> {
    /// Generate a new TLS key with the given destructor.
    /// `max_size` determines the integer size the key has to fit in.
    #[allow(clippy::integer_arithmetic)]
    pub fn create_tls_key(
        &mut self,
        dtor: Option<ty::Instance<'tcx>>,
        max_size: Size,
    ) -> InterpResult<'tcx, TlsKey> {
        let new_key = self.next_key;
        self.next_key += 1;
        self.keys.try_insert(new_key, TlsEntry { data: Default::default(), dtor }).unwrap();
        trace!("New TLS key allocated: {} with dtor {:?}", new_key, dtor);

        if max_size.bits() < 128 && new_key >= (1u128 << max_size.bits()) {
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
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        match self.keys.get(&key) {
            Some(TlsEntry { data, .. }) => {
                let value = data.get(&thread_id).copied();
                trace!("TLS key {} for thread {:?} loaded: {:?}", key, thread_id, value);
                Ok(value.unwrap_or_else(|| Scalar::null_ptr(cx)))
            }
            None => throw_ub_format!("loading from a non-existing TLS key: {}", key),
        }
    }

    pub fn store_tls(
        &mut self,
        key: TlsKey,
        thread_id: ThreadId,
        new_data: Scalar<Provenance>,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx> {
        match self.keys.get_mut(&key) {
            Some(TlsEntry { data, .. }) => {
                if new_data.to_target_usize(cx)? != 0 {
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
    /// NOTE: this does not need locks because it only operates on current thread data
    pub fn set_macos_thread_dtor(
        &mut self,
        thread: ThreadId,
        dtor: ty::Instance<'tcx>,
        data: Scalar<Provenance>,
    ) -> InterpResult<'tcx> {
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
    /// **The order of destructor calls is unspecified if more than one destructor
    /// exists for a thread when it exits.**
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
    ) -> Option<(ty::Instance<'tcx>, Scalar<Provenance>, TlsKey)> {
        use std::ops::Bound::*;

        let thread_local = &mut self.keys;
        let start = match key {
            Some(key) => Excluded(key),
            None => Unbounded,
        };
        // We interpret the documentaion above (taken from POSIX) as saying that we need to iterate
        // over all keys and run each destructor at least once before running any destructor a 2nd
        // time. That's why we have `key` to indicate how far we got in the current iteration. If we
        // return `None`, `schedule_next_pthread_tls_dtor` will re-try with `ket` set to `None` to
        // start the next round.
        // TODO: In the future, we might consider randomizing destructor order, but we still have to
        // uphold this requirement.
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

    /// Delete all TLS entries for the given thread. This function should be
    /// called after all TLS destructors have already finished.
    fn delete_all_thread_tls(&mut self, thread_id: ThreadId) {
        for TlsEntry { data, .. } in self.keys.values_mut() {
            data.remove(&thread_id);
        }
    }
}

impl VisitTags for TlsData<'_> {
    fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
        let TlsData { keys, macos_thread_dtors, next_key: _ } = self;

        for scalar in keys.values().flat_map(|v| v.data.values()) {
            scalar.visit_tags(visit);
        }
        for (_, scalar) in macos_thread_dtors.values() {
            scalar.visit_tags(visit);
        }
    }
}

#[derive(Debug, Default)]
pub struct TlsDtorsState(TlsDtorsStatePriv);

#[derive(Debug, Default)]
enum TlsDtorsStatePriv {
    #[default]
    Init,
    PthreadDtors(RunningDtorState),
    Done,
}

impl TlsDtorsState {
    pub fn on_stack_empty<'tcx>(
        &mut self,
        this: &mut MiriInterpCx<'_, 'tcx>,
    ) -> InterpResult<'tcx, Poll<()>> {
        use TlsDtorsStatePriv::*;
        match &mut self.0 {
            Init => {
                match this.tcx.sess.target.os.as_ref() {
                    "linux" | "freebsd" | "android" => {
                        // Run the pthread dtors.
                        self.0 = PthreadDtors(Default::default());
                    }
                    "macos" => {
                        // The macOS thread wide destructor runs "before any TLS slots get
                        // freed", so do that first.
                        this.schedule_macos_tls_dtor()?;
                        // When the stack is empty again, go on with the pthread dtors.
                        self.0 = PthreadDtors(Default::default());
                    }
                    "windows" => {
                        // Run the special magic hook.
                        this.schedule_windows_tls_dtors()?;
                        // And move to the final state.
                        self.0 = Done;
                    }
                    _ => {
                        // No TLS dtor support.
                        // FIXME: should we do something on wasi?
                        self.0 = Done;
                    }
                }
            }
            PthreadDtors(state) => {
                match this.schedule_next_pthread_tls_dtor(state)? {
                    Poll::Pending => {} // just keep going
                    Poll::Ready(()) => self.0 = Done,
                }
            }
            Done => {
                this.machine.tls.delete_all_thread_tls(this.get_active_thread());
                return Ok(Poll::Ready(()));
            }
        }

        Ok(Poll::Pending)
    }
}

impl<'mir, 'tcx: 'mir> EvalContextPrivExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
trait EvalContextPrivExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    /// Schedule TLS destructors for Windows.
    /// On windows, TLS destructors are managed by std.
    fn schedule_windows_tls_dtors(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // Windows has a special magic linker section that is run on certain events.
        // Instead of searching for that section and supporting arbitrary hooks in there
        // (that would be basically https://github.com/rust-lang/miri/issues/450),
        // we specifically look up the static in libstd that we know is placed
        // in that section.
        if !this.have_module(&["std"]) {
            // Looks like we are running in a `no_std` crate.
            // That also means no TLS dtors callback to call.
            return Ok(());
        }
        let thread_callback =
            this.eval_windows("thread_local_key", "p_thread_callback").to_pointer(this)?;
        let thread_callback = this.get_ptr_fn(thread_callback)?.as_instance()?;

        // FIXME: Technically, the reason should be `DLL_PROCESS_DETACH` when the main thread exits
        // but std treats both the same.
        let reason = this.eval_windows("c", "DLL_THREAD_DETACH");

        // The signature of this function is `unsafe extern "system" fn(h: c::LPVOID, dwReason: c::DWORD, pv: c::LPVOID)`.
        // FIXME: `h` should be a handle to the current module and what `pv` should be is unknown
        // but both are ignored by std
        this.call_function(
            thread_callback,
            Abi::System { unwind: false },
            &[Scalar::null_ptr(this).into(), reason.into(), Scalar::null_ptr(this).into()],
            None,
            StackPopCleanup::Root { cleanup: true },
        )?;
        Ok(())
    }

    /// Schedule the MacOS thread destructor of the thread local storage to be
    /// executed.
    fn schedule_macos_tls_dtor(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let thread_id = this.get_active_thread();
        if let Some((instance, data)) = this.machine.tls.macos_thread_dtors.remove(&thread_id) {
            trace!("Running macos dtor {:?} on {:?} at {:?}", instance, data, thread_id);

            this.call_function(
                instance,
                Abi::C { unwind: false },
                &[data.into()],
                None,
                StackPopCleanup::Root { cleanup: true },
            )?;
        }
        Ok(())
    }

    /// Schedule a pthread TLS destructor. Returns `true` if found
    /// a destructor to schedule, and `false` otherwise.
    fn schedule_next_pthread_tls_dtor(
        &mut self,
        state: &mut RunningDtorState,
    ) -> InterpResult<'tcx, Poll<()>> {
        let this = self.eval_context_mut();
        let active_thread = this.get_active_thread();

        // Fetch next dtor after `key`.
        let dtor = match this.machine.tls.fetch_tls_dtor(state.last_key, active_thread) {
            dtor @ Some(_) => dtor,
            // We ran each dtor once, start over from the beginning.
            None => this.machine.tls.fetch_tls_dtor(None, active_thread),
        };
        if let Some((instance, ptr, key)) = dtor {
            state.last_key = Some(key);
            trace!("Running TLS dtor {:?} on {:?} at {:?}", instance, ptr, active_thread);
            assert!(
                !ptr.to_target_usize(this).unwrap() != 0,
                "data can't be NULL when dtor is called!"
            );

            this.call_function(
                instance,
                Abi::C { unwind: false },
                &[ptr.into()],
                None,
                StackPopCleanup::Root { cleanup: true },
            )?;

            return Ok(Poll::Pending);
        }

        Ok(Poll::Ready(()))
    }
}
