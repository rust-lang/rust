//! Implement thread-local storage.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry as BTreeEntry;
use std::task::Poll;

use rustc_abi::{ExternAbi, HasDataLayout, Size};
use rustc_middle::ty;

use crate::*;

pub type TlsKey = u128;

#[derive(Clone, Debug)]
pub struct TlsEntry<'tcx> {
    /// The data for this key. None is used to represent NULL.
    /// (We normalize this early to avoid having to do a NULL-ptr-test each time we access the data.)
    data: BTreeMap<ThreadId, Scalar>,
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

    /// On macOS, each thread holds a list of destructor functions with their
    /// respective data arguments.
    macos_thread_dtors: BTreeMap<ThreadId, Vec<(ty::Instance<'tcx>, Scalar)>>,
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
    #[expect(clippy::arithmetic_side_effects)]
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
        interp_ok(new_key)
    }

    pub fn delete_tls_key(&mut self, key: TlsKey) -> InterpResult<'tcx> {
        match self.keys.remove(&key) {
            Some(_) => {
                trace!("TLS key {} removed", key);
                interp_ok(())
            }
            None => throw_ub_format!("removing a nonexistent TLS key: {}", key),
        }
    }

    pub fn load_tls(
        &self,
        key: TlsKey,
        thread_id: ThreadId,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx, Scalar> {
        match self.keys.get(&key) {
            Some(TlsEntry { data, .. }) => {
                let value = data.get(&thread_id).copied();
                trace!("TLS key {} for thread {:?} loaded: {:?}", key, thread_id, value);
                interp_ok(value.unwrap_or_else(|| Scalar::null_ptr(cx)))
            }
            None => throw_ub_format!("loading from a non-existing TLS key: {}", key),
        }
    }

    pub fn store_tls(
        &mut self,
        key: TlsKey,
        thread_id: ThreadId,
        new_data: Scalar,
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
                interp_ok(())
            }
            None => throw_ub_format!("storing to a non-existing TLS key: {}", key),
        }
    }

    /// Add a thread local storage destructor for the given thread. This function
    /// is used to implement the `_tlv_atexit` shim on MacOS.
    pub fn add_macos_thread_dtor(
        &mut self,
        thread: ThreadId,
        dtor: ty::Instance<'tcx>,
        data: Scalar,
    ) -> InterpResult<'tcx> {
        self.macos_thread_dtors.entry(thread).or_default().push((dtor, data));
        interp_ok(())
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
    ) -> Option<(ty::Instance<'tcx>, Scalar, TlsKey)> {
        use std::ops::Bound::*;

        let thread_local = &mut self.keys;
        let start = match key {
            Some(key) => Excluded(key),
            None => Unbounded,
        };
        // We interpret the documentation above (taken from POSIX) as saying that we need to iterate
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

        if let Some(dtors) = self.macos_thread_dtors.remove(&thread_id) {
            assert!(dtors.is_empty(), "the destructors should have already been run");
        }
    }
}

impl VisitProvenance for TlsData<'_> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        let TlsData { keys, macos_thread_dtors, next_key: _ } = self;

        for scalar in keys.values().flat_map(|v| v.data.values()) {
            scalar.visit_provenance(visit);
        }
        for (_, scalar) in macos_thread_dtors.values().flatten() {
            scalar.visit_provenance(visit);
        }
    }
}

#[derive(Debug, Default)]
pub struct TlsDtorsState<'tcx>(TlsDtorsStatePriv<'tcx>);

#[derive(Debug, Default)]
enum TlsDtorsStatePriv<'tcx> {
    #[default]
    Init,
    MacOsDtors,
    PthreadDtors(RunningDtorState),
    /// For Windows Dtors, we store the list of functions that we still have to call.
    /// These are functions from the magic `.CRT$XLB` linker section.
    WindowsDtors(Vec<ImmTy<'tcx>>),
    Done,
}

impl<'tcx> TlsDtorsState<'tcx> {
    pub fn on_stack_empty(
        &mut self,
        this: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Poll<()>> {
        use TlsDtorsStatePriv::*;
        let new_state = 'new_state: {
            match &mut self.0 {
                Init => {
                    match this.tcx.sess.target.os.as_ref() {
                        "macos" => {
                            // macOS has a _tlv_atexit function that allows
                            // registering destructors without associated keys.
                            // These are run first.
                            break 'new_state MacOsDtors;
                        }
                        _ if this.target_os_is_unix() => {
                            // All other Unixes directly jump to running the pthread dtors.
                            break 'new_state PthreadDtors(Default::default());
                        }
                        "windows" => {
                            // Determine which destructors to run.
                            let dtors = this.lookup_windows_tls_dtors()?;
                            // And move to the next state, that runs them.
                            break 'new_state WindowsDtors(dtors);
                        }
                        _ => {
                            // No TLS dtor support.
                            // FIXME: should we do something on wasi?
                            break 'new_state Done;
                        }
                    }
                }
                MacOsDtors => {
                    match this.schedule_macos_tls_dtor()? {
                        Poll::Pending => return interp_ok(Poll::Pending),
                        // After all macOS destructors are run, the system switches
                        // to destroying the pthread destructors.
                        Poll::Ready(()) => break 'new_state PthreadDtors(Default::default()),
                    }
                }
                PthreadDtors(state) => {
                    match this.schedule_next_pthread_tls_dtor(state)? {
                        Poll::Pending => return interp_ok(Poll::Pending), // just keep going
                        Poll::Ready(()) => break 'new_state Done,
                    }
                }
                WindowsDtors(dtors) => {
                    if let Some(dtor) = dtors.pop() {
                        this.schedule_windows_tls_dtor(dtor)?;
                        return interp_ok(Poll::Pending); // we stay in this state (but `dtors` got shorter)
                    } else {
                        // No more destructors to run.
                        break 'new_state Done;
                    }
                }
                Done => {
                    this.machine.tls.delete_all_thread_tls(this.active_thread());
                    return interp_ok(Poll::Ready(()));
                }
            }
        };

        self.0 = new_state;
        interp_ok(Poll::Pending)
    }
}

impl<'tcx> EvalContextPrivExt<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Schedule TLS destructors for Windows.
    /// On windows, TLS destructors are managed by std.
    fn lookup_windows_tls_dtors(&mut self) -> InterpResult<'tcx, Vec<ImmTy<'tcx>>> {
        let this = self.eval_context_mut();

        // Windows has a special magic linker section that is run on certain events.
        // We don't support most of that, but just enough to make thread-local dtors in `std` work.
        interp_ok(this.lookup_link_section(|section| section == ".CRT$XLB")?)
    }

    fn schedule_windows_tls_dtor(&mut self, dtor: ImmTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let dtor = dtor.to_scalar().to_pointer(this)?;
        let thread_callback = this.get_ptr_fn(dtor)?.as_instance()?;

        // FIXME: Technically, the reason should be `DLL_PROCESS_DETACH` when the main thread exits
        // but std treats both the same.
        let reason = this.eval_windows("c", "DLL_THREAD_DETACH");
        let null_ptr =
            ImmTy::from_scalar(Scalar::null_ptr(this), this.machine.layouts.const_raw_ptr);

        // The signature of this function is `unsafe extern "system" fn(h: c::LPVOID, dwReason: c::DWORD, pv: c::LPVOID)`.
        // FIXME: `h` should be a handle to the current module and what `pv` should be is unknown
        // but both are ignored by std.
        this.call_function(
            thread_callback,
            ExternAbi::System { unwind: false },
            &[null_ptr.clone(), ImmTy::from_scalar(reason, this.machine.layouts.u32), null_ptr],
            None,
            ReturnContinuation::Stop { cleanup: true },
        )?;
        interp_ok(())
    }

    /// Schedule the macOS thread local storage destructors to be executed.
    fn schedule_macos_tls_dtor(&mut self) -> InterpResult<'tcx, Poll<()>> {
        let this = self.eval_context_mut();
        let thread_id = this.active_thread();
        // macOS keeps track of TLS destructors in a stack. If a destructor
        // registers another destructor, it will be run next.
        // See https://github.com/apple-oss-distributions/dyld/blob/d552c40cd1de105f0ec95008e0e0c0972de43456/dyld/DyldRuntimeState.cpp#L2277
        let dtor = this.machine.tls.macos_thread_dtors.get_mut(&thread_id).and_then(Vec::pop);
        if let Some((instance, data)) = dtor {
            trace!("Running macos dtor {:?} on {:?} at {:?}", instance, data, thread_id);

            this.call_function(
                instance,
                ExternAbi::C { unwind: false },
                &[ImmTy::from_scalar(data, this.machine.layouts.mut_raw_ptr)],
                None,
                ReturnContinuation::Stop { cleanup: true },
            )?;

            return interp_ok(Poll::Pending);
        }

        interp_ok(Poll::Ready(()))
    }

    /// Schedule a pthread TLS destructor. Returns `true` if found
    /// a destructor to schedule, and `false` otherwise.
    fn schedule_next_pthread_tls_dtor(
        &mut self,
        state: &mut RunningDtorState,
    ) -> InterpResult<'tcx, Poll<()>> {
        let this = self.eval_context_mut();
        let active_thread = this.active_thread();

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
                ptr.to_target_usize(this).unwrap() != 0,
                "data can't be NULL when dtor is called!"
            );

            this.call_function(
                instance,
                ExternAbi::C { unwind: false },
                &[ImmTy::from_scalar(ptr, this.machine.layouts.mut_raw_ptr)],
                None,
                ReturnContinuation::Stop { cleanup: true },
            )?;

            return interp_ok(Poll::Pending);
        }

        interp_ok(Poll::Ready(()))
    }
}
