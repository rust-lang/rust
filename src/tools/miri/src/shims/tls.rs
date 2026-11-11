//! Implement thread-local storage.

use std::collections::btree_map::Entry as BTreeEntry;
use std::collections::{BTreeMap, VecDeque};
use std::task::Poll;

use rustc_abi::{ExternAbi, Size};
use rustc_middle::ty;
use rustc_span::Span;
use rustc_target::spec::Os;

use crate::*;

pub type TlsKey = u128;

#[derive(Clone, Debug)]
pub struct TlsEntry<'tcx> {
    /// The data for this key. Absence of data for a thread ID is used to represent NULL.
    /// (We normalize this early to avoid having to do a NULL-ptr-test each time we access the data.)
    pub data: BTreeMap<ThreadId, Pointer>,
    pub dtor: Option<(ty::Instance<'tcx>, Span)>,
}

#[derive(Default, Debug)]
struct RunningPthreadDtorState {
    /// The last TlsKey used to retrieve a TLS destructor. `None` means that we
    /// have not tried to retrieve a TLS destructor yet or that we already tried
    /// all keys.
    last_key: Option<TlsKey>,
}

#[derive(Default, Debug)]
struct RunningWindowsDtorState {
    /// The last TlsKey for which a TLS (really, FLS) destructor ran. `None` means that we
    /// have not run a TLS destructor yet. This is used to clear the TLS value after the dtor returned.
    last_key: Option<TlsKey>,
    // Keys that have destructors that we still need to run.
    remaining_keys: VecDeque<TlsKey>,
}

#[derive(Debug)]
pub struct TlsData<'tcx> {
    /// The Key to use for the next thread-local allocation.
    next_key: TlsKey,

    /// pthreads-style thread-local storage.
    keys: BTreeMap<TlsKey, TlsEntry<'tcx>>,

    /// Linux and macOS allow registering callbacks that are run when the current thread exits.
    thread_atexit: BTreeMap<ThreadId, Vec<(ty::Instance<'tcx>, Scalar, Span)>>,
}

impl<'tcx> Default for TlsData<'tcx> {
    fn default() -> Self {
        TlsData {
            next_key: 1, // start with 1 as we must not use 0 on Windows
            keys: Default::default(),
            thread_atexit: Default::default(),
        }
    }
}

impl<'tcx> TlsData<'tcx> {
    /// Generate a new TLS key with the given destructor.
    /// `key_size` determines the integer size the key has to fit in.
    #[expect(clippy::arithmetic_side_effects)]
    pub fn create_tls_key(
        &mut self,
        dtor: Option<(ty::Instance<'tcx>, Span)>,
        key_size: Size,
    ) -> InterpResult<'tcx, TlsKey> {
        let new_key = self.next_key;
        self.next_key += 1;
        self.keys.try_insert(new_key, TlsEntry { data: Default::default(), dtor }).unwrap();
        trace!("New TLS key allocated: {} with dtor {:?}", new_key, dtor);

        if new_key > key_size.unsigned_int_max() {
            throw_unsup_format!("we ran out of TLS key space");
        }
        interp_ok(new_key)
    }

    pub fn delete_tls_key(&mut self, key: TlsKey) -> InterpResult<'tcx, TlsEntry<'tcx>> {
        match self.keys.remove(&key) {
            Some(entry) => {
                trace!("TLS key {} removed", key);
                interp_ok(entry)
            }
            None => throw_ub_format!("removing a nonexistent TLS key: {}", key),
        }
    }

    pub fn load_tls_key(&self, key: TlsKey, thread_id: ThreadId) -> InterpResult<'tcx, Pointer> {
        match self.keys.get(&key) {
            Some(TlsEntry { data, .. }) => {
                let value = data.get(&thread_id).copied();
                trace!("TLS key {} for thread {:?} loaded: {:?}", key, thread_id, value);
                interp_ok(value.unwrap_or_else(Pointer::null))
            }
            None => throw_ub_format!("loading from a non-existing TLS key: {}", key),
        }
    }

    pub fn store_tls_key(
        &mut self,
        key: TlsKey,
        thread_id: ThreadId,
        new_data: Pointer,
    ) -> InterpResult<'tcx> {
        match self.keys.get_mut(&key) {
            Some(TlsEntry { data, .. }) => {
                if new_data.addr() != Size::ZERO {
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
    pub fn add_thread_atexit(
        &mut self,
        thread: ThreadId,
        dtor: ty::Instance<'tcx>,
        data: Scalar,
        span: Span,
    ) -> InterpResult<'tcx> {
        self.thread_atexit.entry(thread).or_default().push((dtor, data, span));
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
    fn fetch_tls_key_dtor(
        &mut self,
        key: Option<TlsKey>,
        thread_id: ThreadId,
    ) -> Option<(ty::Instance<'tcx>, Pointer, TlsKey, Span)> {
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
                    if let Some((dtor, span)) = dtor {
                        // Set TLS data to NULL, and call dtor with old value.
                        let data = entry.remove();
                        return Some((*dtor, data, key, *span));
                    }
                }
                BTreeEntry::Vacant(_) => {}
            }
        }
        None
    }

    /// Delete all TLS entries for the given thread. This function should be
    /// called after all TLS destructors have already finished.
    fn delete_all_thread_tls_keys(&mut self, thread_id: ThreadId) {
        for TlsEntry { data, .. } in self.keys.values_mut() {
            data.remove(&thread_id);
        }

        if let Some(dtors) = self.thread_atexit.remove(&thread_id) {
            assert!(dtors.is_empty(), "the destructors should have already been run");
        }
    }

    pub fn leak_all_thread_tls_keys(&self, thread_id: ThreadId, static_roots: &mut Vec<AllocId>) {
        for TlsEntry { data, .. } in self.keys.values() {
            if let Some(ptr) = data.get(&thread_id) {
                if let Some(alloc) = ptr.provenance.and_then(|p| p.get_alloc_id()) {
                    trace!("Thread-local key value leaked and stored as static root: {:?}", alloc);
                    static_roots.push(alloc);
                }
            }
        }
    }
}

impl VisitProvenance for TlsData<'_> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        let TlsData { keys, thread_atexit: macos_thread_dtors, next_key: _ } = self;

        for scalar in keys.values().flat_map(|v| v.data.values()) {
            scalar.visit_provenance(visit);
        }
        for (_, scalar, _) in macos_thread_dtors.values().flatten() {
            scalar.visit_provenance(visit);
        }
    }
}

#[derive(Debug)]
pub struct TlsDtorsState<'tcx> {
    state: TlsDtorsStatePriv<'tcx>,
    /// Whether this is for program exit vs thread finish.
    for_exit: bool,
}

#[derive(Debug)]
enum TlsDtorsStatePriv<'tcx> {
    Init,
    /// Running native thread destructors (`TlsData::thread_atexit`).
    ThreadAtexit,
    /// Running pthread key destructors.
    PthreadDtors(RunningPthreadDtorState),
    /// For Windows, we support two different ways dtors can be registered.
    /// 1. Functions that are registered via the `FlsAlloc` function, which are invoked one by one.
    /// 2. Functions from the magic `.CRT$XLB` linker section.
    ///    We store these as a list of functions that we still have to call.
    WindowsDtors(RunningWindowsDtorState, Vec<(ImmTy<'tcx>, Span)>),
    Done,
}

impl<'tcx> TlsDtorsState<'tcx> {
    /// Returns the state we should start running when the program is exiting.
    pub fn new_for_exit() -> Self {
        Self { state: TlsDtorsStatePriv::Init, for_exit: true }
    }

    /// Returns the state we should start running when a thread finished running.
    pub fn new_for_thread_finish() -> Self {
        Self { state: TlsDtorsStatePriv::Init, for_exit: false }
    }

    pub fn on_stack_empty(&mut self, ecx: &mut MiriInterpCx<'tcx>) -> InterpResult<'tcx, Poll<()>> {
        use TlsDtorsStatePriv::*;
        let new_state = 'new_state: {
            match &mut self.state {
                Init => {
                    break 'new_state ThreadAtexit;
                }
                ThreadAtexit => {
                    let Poll::Ready(()) = ecx.schedule_thread_atexit()? else {
                        return interp_ok(Poll::Pending);
                    };
                    // Native thread_atexit handlers are done.
                    // What's next? Depends on the target!
                    match ecx.tcx.sess.target.os {
                        _ if ecx.target_os_is_unix() => {
                            // On thread finish, we run pthread dtors. But not on exit!
                            if self.for_exit {
                                // Leak pthread dtor contents, like thread-local statics.
                                ecx.machine.tls.leak_all_thread_tls_keys(
                                    ecx.active_thread(),
                                    &mut ecx.machine.static_roots,
                                );
                                break 'new_state Done;
                            } else {
                                break 'new_state PthreadDtors(Default::default());
                            }
                        }
                        Os::Windows => {
                            // Determine which destructors to run.
                            let dtors = ecx.lookup_windows_tls_dtors()?;

                            // Fetch fls keys that have destructors. Keys registered during thread
                            // exit may or may not have their destructor called. We decide not to
                            // call them. See also: `schedule_next_windows_fls_dtor`.
                            let fls_keys_with_dtors = ecx.lookup_windows_fls_keys_with_dtors()?;

                            // And move to the next state, that runs them.
                            break 'new_state WindowsDtors(
                                RunningWindowsDtorState {
                                    last_key: None,
                                    remaining_keys: fls_keys_with_dtors,
                                },
                                dtors,
                            );
                        }
                        _ => {
                            break 'new_state Done;
                        }
                    }
                }
                PthreadDtors(state) => {
                    let Poll::Ready(()) = ecx.schedule_next_pthread_tls_dtor(state)? else {
                        return interp_ok(Poll::Pending);
                    };
                    break 'new_state Done;
                }
                WindowsDtors(state, dtors) => {
                    // Fls destructors are scheduled before the tls callback.
                    let Poll::Ready(()) = ecx.schedule_next_windows_fls_dtor(state)? else {
                        return interp_ok(Poll::Pending);
                    };

                    // See if there's another callback to run
                    if let Some((dtor, span)) = dtors.pop() {
                        ecx.schedule_windows_tls_dtor(dtor, span)?;
                        return interp_ok(Poll::Pending); // we stay in this state (but `dtors` got shorter)
                    } else {
                        // No more destructors to run.
                        break 'new_state Done;
                    }
                }
                Done => {
                    // All destructors are called. Clean up remaining keys so they don't linger in
                    // the interpreter state.
                    ecx.machine.tls.delete_all_thread_tls_keys(ecx.active_thread());
                    return interp_ok(Poll::Ready(()));
                }
            }
        };

        self.state = new_state;
        interp_ok(Poll::Pending)
    }
}

impl<'tcx> EvalContextPrivExt<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Schedule TLS destructors for Windows.
    /// On windows, TLS destructors are managed by std.
    fn lookup_windows_tls_dtors(&mut self) -> InterpResult<'tcx, Vec<(ImmTy<'tcx>, Span)>> {
        let ecx = self.eval_context_mut();

        // Windows has a special magic linker section that is run on certain events.
        // We don't support most of that, but just enough to make thread-local dtors in `std` work.
        interp_ok(ecx.lookup_link_section(|section| section == ".CRT$XLB")?)
    }

    /// Lookup all the FLS keys (which are stored as TLS keys) that have a destructor.
    /// See also: `schedule_next_windows_fls_dtor`.
    fn lookup_windows_fls_keys_with_dtors(&self) -> InterpResult<'tcx, VecDeque<TlsKey>> {
        let ecx = self.eval_context_ref();

        interp_ok(
            ecx.machine
                .tls
                .keys
                .iter()
                .filter(|(_, data)| data.dtor.is_some())
                .map(|(key, _)| *key)
                .collect(),
        )
    }

    fn schedule_windows_tls_dtor(&mut self, dtor: ImmTy<'tcx>, span: Span) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        let dtor = dtor.to_scalar().to_pointer(ecx);
        let thread_callback = ecx.get_ptr_fn(dtor)?.as_instance()?;

        // FIXME: Technically, the reason should be `DLL_PROCESS_DETACH` when the main thread exits
        // but std treats both the same.
        let reason = ecx.eval_windows("c", "DLL_THREAD_DETACH");
        let null_ptr =
            ImmTy::from_scalar(Scalar::null_ptr(ecx), ecx.machine.layouts.unit_ptr_const);

        // The signature of this function is `unsafe extern "system" fn(h: c::LPVOID, dwReason: c::DWORD, pv: c::LPVOID)`.
        // FIXME: `h` should be a handle to the current module and what `pv` should be is unknown
        // but both are ignored by std.
        ecx.call_thread_root_function(
            thread_callback,
            ExternAbi::System { unwind: false },
            &[null_ptr.clone(), ImmTy::from_scalar(reason, ecx.machine.layouts.u32), null_ptr],
            None,
            span,
        )?;
        interp_ok(())
    }

    /// Schedule the macOS thread local storage destructors to be executed.
    fn schedule_thread_atexit(&mut self) -> InterpResult<'tcx, Poll<()>> {
        let ecx = self.eval_context_mut();
        let thread_id = ecx.active_thread();
        // macOS keeps track of TLS destructors in a stack. If a destructor
        // registers another destructor, it will be run next.
        // See https://github.com/apple-oss-distributions/dyld/blob/d552c40cd1de105f0ec95008e0e0c0972de43456/dyld/DyldRuntimeState.cpp#L2277
        let dtor = ecx.machine.tls.thread_atexit.get_mut(&thread_id).and_then(Vec::pop);
        if let Some((instance, data, span)) = dtor {
            trace!("Running thread_atexit hook {:?} on {:?} at {:?}", instance, data, thread_id);

            ecx.call_thread_root_function(
                instance,
                ExternAbi::C { unwind: false },
                &[ImmTy::from_scalar(data, ecx.machine.layouts.unit_ptr_mut)],
                None,
                span,
            )?;

            return interp_ok(Poll::Pending);
        }

        interp_ok(Poll::Ready(()))
    }

    /// Schedule a pthread TLS destructor. Returns `true` if found
    /// a destructor to schedule, and `false` otherwise.
    fn schedule_next_pthread_tls_dtor(
        &mut self,
        state: &mut RunningPthreadDtorState,
    ) -> InterpResult<'tcx, Poll<()>> {
        let ecx = self.eval_context_mut();
        let active_thread = ecx.active_thread();

        // Fetch next dtor after `key`.
        let dtor = match ecx.machine.tls.fetch_tls_key_dtor(state.last_key, active_thread) {
            dtor @ Some(_) => dtor,
            // We ran each dtor once, start over from the beginning.
            None => ecx.machine.tls.fetch_tls_key_dtor(None, active_thread),
        };
        if let Some((instance, ptr, key, span)) = dtor {
            state.last_key = Some(key);
            trace!("Running TLS dtor {:?} on {:?} at {:?}", instance, ptr, active_thread);
            assert!(ptr.addr() != Size::ZERO, "data can't be NULL when dtor is called!");

            ecx.call_thread_root_function(
                instance,
                ExternAbi::C { unwind: false },
                &[ImmTy::from_scalar(
                    Scalar::from_maybe_pointer(ptr, ecx),
                    ecx.machine.layouts.unit_ptr_mut,
                )],
                None,
                span,
            )?;

            return interp_ok(Poll::Pending);
        }

        interp_ok(Poll::Ready(()))
    }

    /// Schedule a Windows FLS destructor, if one is found.
    fn schedule_next_windows_fls_dtor(
        &mut self,
        state: &mut RunningWindowsDtorState,
    ) -> InterpResult<'tcx, Poll<()>> {
        let ecx = self.eval_context_mut();
        let active_thread = ecx.active_thread();

        // According to [PflsCallbackFunction's docs],
        // > If the FLS slot is in use, `FlsCallback`` is called on .. thread exit ..
        // However, the exact order and semantics are not defined.
        // We use the following implementation, which matches both observed behavior and
        // Wine's implementation of the same logic in the [`RtlProcessFlsData`] function.
        // 1. Fetch all the keys once (in `lookup_windows_fls_keys_with_dtors`).
        // 2. Go over them one by one, in order, skipping keys without dtors.
        // 3. Fetch the value associated with the key.
        // 4. If it is non-zero, call the registered dtor.
        // 5. After the dtor is called, clear the key's value, setting it to zero.
        // New keys registered during thread exit can be ignored (but sometimes they do run
        // natively, just not reliably), but values set before the dtor is scheduled are visible.
        // Keys without dtors will not be set to zero.
        // [PflsCallbackFunction's docs]: https://learn.microsoft.com/en-us/windows/win32/api/winnt/nc-winnt-pfls_callback_function
        // [`RtlProcessFlsData`]: https://github.com/wine-mirror/wine/blob/wine-11.0/dlls/ntdll/thread.c#L679

        // We are done running the previous key's destructor, so set it's value to zero.
        if let Some(last_key) = state.last_key.take() {
            if let Some(TlsEntry { data, .. }) = ecx.machine.tls.keys.get_mut(&last_key) {
                data.remove(&active_thread);
            };
        }

        while let Some(key) = state.remaining_keys.pop_front() {
            // Fetch dtor for this `key`.
            // If the key doesn't have a dtor or does not exist any more, move on to the next key.
            let Some(TlsEntry { data, dtor: Some(dtor) }) = ecx.machine.tls.keys.get(&key) else {
                continue;
            };

            let (instance, span) = dtor.to_owned();

            // If the key has no value in this thread, move on to the next key.
            let Some(&ptr) = data.get(&active_thread) else { continue };

            assert!(
                ptr.addr() != Size::ZERO,
                "TLS key's value can't be null (should be absent instead)"
            );

            trace!("Running TLS dtor {:?} on {:?} at {:?}", instance, ptr, active_thread);

            // We'll clear this key's value next time we are called.
            state.last_key = Some(key);

            ecx.call_thread_root_function(
                instance,
                ExternAbi::System { unwind: false },
                &[ImmTy::from_scalar(
                    Scalar::from_maybe_pointer(ptr, ecx),
                    ecx.machine.layouts.unit_ptr_mut,
                )],
                None,
                span,
            )?;

            return interp_ok(Poll::Pending);
        }

        // We are done scheduling all the keys.
        interp_ok(Poll::Ready(()))
    }
}
