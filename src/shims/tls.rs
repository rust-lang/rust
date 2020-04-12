//! Implement thread-local storage.

use std::collections::BTreeMap;

use log::trace;

use rustc_middle::ty;
use rustc_target::abi::{LayoutOf, Size, HasDataLayout};

use crate::{HelpersEvalContextExt, InterpResult, MPlaceTy, Scalar, StackPopCleanup, Tag};

pub type TlsKey = u128;

#[derive(Copy, Clone, Debug)]
pub struct TlsEntry<'tcx> {
    /// The data for this key. None is used to represent NULL.
    /// (We normalize this early to avoid having to do a NULL-ptr-test each time we access the data.)
    /// Will eventually become a map from thread IDs to `Scalar`s, if we ever support more than one thread.
    data: Option<Scalar<Tag>>,
    dtor: Option<ty::Instance<'tcx>>,
}

#[derive(Debug)]
pub struct TlsData<'tcx> {
    /// The Key to use for the next thread-local allocation.
    next_key: TlsKey,

    /// pthreads-style thread-local storage.
    keys: BTreeMap<TlsKey, TlsEntry<'tcx>>,

    /// A single global dtor (that's how things work on macOS) with a data argument.
    global_dtor: Option<(ty::Instance<'tcx>, Scalar<Tag>)>,

    /// Whether we are in the "destruct" phase, during which some operations are UB.
    dtors_running: bool,
}

impl<'tcx> Default for TlsData<'tcx> {
    fn default() -> Self {
        TlsData {
            next_key: 1, // start with 1 as we must not use 0 on Windows
            keys: Default::default(),
            global_dtor: None,
            dtors_running: false,
        }
    }
}

impl<'tcx> TlsData<'tcx> {
    /// Generate a new TLS key with the given destructor.
    /// `max_size` determines the integer size the key has to fit in.
    pub fn create_tls_key(&mut self, dtor: Option<ty::Instance<'tcx>>, max_size: Size) -> InterpResult<'tcx, TlsKey> {
        let new_key = self.next_key;
        self.next_key += 1;
        self.keys.insert(new_key, TlsEntry { data: None, dtor }).unwrap_none();
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
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx, Scalar<Tag>> {
        match self.keys.get(&key) {
            Some(&TlsEntry { data, .. }) => {
                trace!("TLS key {} loaded: {:?}", key, data);
                Ok(data.unwrap_or_else(|| Scalar::null_ptr(cx).into()))
            }
            None => throw_ub_format!("loading from a non-existing TLS key: {}", key),
        }
    }

    pub fn store_tls(&mut self, key: TlsKey, new_data: Option<Scalar<Tag>>) -> InterpResult<'tcx> {
        match self.keys.get_mut(&key) {
            Some(TlsEntry { data, .. }) => {
                trace!("TLS key {} stored: {:?}", key, new_data);
                *data = new_data;
                Ok(())
            }
            None => throw_ub_format!("storing to a non-existing TLS key: {}", key),
        }
    }

    pub fn set_global_dtor(&mut self, dtor: ty::Instance<'tcx>, data: Scalar<Tag>) -> InterpResult<'tcx> {
        if self.dtors_running {
            // UB, according to libstd docs.
            throw_ub_format!("setting global destructor while destructors are already running");
        }
        if self.global_dtor.is_some() {
            throw_unsup_format!("setting more than one global destructor is not supported");
        }

        self.global_dtor = Some((dtor, data));
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
    ) -> Option<(ty::Instance<'tcx>, Scalar<Tag>, TlsKey)> {
        use std::collections::Bound::*;

        let thread_local = &mut self.keys;
        let start = match key {
            Some(key) => Excluded(key),
            None => Unbounded,
        };
        for (&key, TlsEntry { data, dtor }) in
            thread_local.range_mut((start, Unbounded))
        {
            if let Some(data_scalar) = *data {
                if let Some(dtor) = dtor {
                    let ret = Some((*dtor, data_scalar, key));
                    *data = None;
                    return ret;
                }
            }
        }
        None
    }
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn run_tls_dtors(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        assert!(!this.machine.tls.dtors_running, "running TLS dtors twice");
        this.machine.tls.dtors_running = true;

        if this.tcx.sess.target.target.target_os == "windows" {
            // Windows has a special magic linker section that is run on certain events.
            // Instead of searching for that section and supporting arbitrary hooks in there
            // (that would be basically https://github.com/rust-lang/miri/issues/450),
            // we specifically look up the static in libstd that we know is placed
            // in that section.
            let thread_callback = this.eval_path_scalar(&["std", "sys", "windows", "thread_local", "p_thread_callback"])?;
            let thread_callback = this.memory.get_fn(thread_callback.not_undef()?)?.as_instance()?;

            // The signature of this function is `unsafe extern "system" fn(h: c::LPVOID, dwReason: c::DWORD, pv: c::LPVOID)`.
            let reason = this.eval_path_scalar(&["std", "sys", "windows", "c", "DLL_PROCESS_DETACH"])?;
            let ret_place = MPlaceTy::dangling(this.layout_of(this.tcx.mk_unit())?, this).into();
            this.call_function(
                thread_callback,
                &[Scalar::null_ptr(this).into(), reason.into(), Scalar::null_ptr(this).into()],
                Some(ret_place),
                StackPopCleanup::None { cleanup: true },
            )?;

            // step until out of stackframes
            this.run()?;

            // Windows doesn't have other destructors.
            return Ok(());
        }

        // The macOS global dtor runs "before any TLS slots get freed", so do that first.
        if let Some((instance, data)) = this.machine.tls.global_dtor {
            trace!("Running global dtor {:?} on {:?}", instance, data);

            let ret_place = MPlaceTy::dangling(this.layout_of(this.tcx.mk_unit())?, this).into();
            this.call_function(
                instance,
                &[data.into()],
                Some(ret_place),
                StackPopCleanup::None { cleanup: true },
            )?;

            // step until out of stackframes
            this.run()?;
        }

        // Now run the "keyed" destructors.
        let mut dtor = this.machine.tls.fetch_tls_dtor(None);
        while let Some((instance, ptr, key)) = dtor {
            trace!("Running TLS dtor {:?} on {:?}", instance, ptr);
            assert!(!this.is_null(ptr).unwrap(), "data can't be NULL when dtor is called!");

            let ret_place = MPlaceTy::dangling(this.layout_of(this.tcx.mk_unit())?, this).into();
            this.call_function(
                instance,
                &[ptr.into()],
                Some(ret_place),
                StackPopCleanup::None { cleanup: true },
            )?;

            // step until out of stackframes
            this.run()?;

            // Fetch next dtor after `key`.
            dtor = match this.machine.tls.fetch_tls_dtor(Some(key)) {
                dtor @ Some(_) => dtor,
                // We ran each dtor once, start over from the beginning.
                None => this.machine.tls.fetch_tls_dtor(None),
            };
        }
        Ok(())
    }
}
