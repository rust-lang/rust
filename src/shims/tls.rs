//! Implement thread-local storage.

use std::collections::BTreeMap;

use rustc_target::abi::LayoutOf;
use rustc::{ty, ty::layout::HasDataLayout};

use crate::{
    InterpResult, StackPopCleanup,
    MPlaceTy, Scalar, Tag,
    HelpersEvalContextExt,
};

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
}

impl<'tcx> Default for TlsData<'tcx> {
    fn default() -> Self {
        TlsData {
            next_key: 1, // start with 1 as we must not use 0 on Windows
            keys: Default::default(),
        }
    }
}

impl<'tcx> TlsData<'tcx> {
    pub fn create_tls_key(
        &mut self,
        dtor: Option<ty::Instance<'tcx>>,
    ) -> TlsKey {
        let new_key = self.next_key;
        self.next_key += 1;
        self.keys.insert(
            new_key,
            TlsEntry {
                data: None,
                dtor,
            },
        ).unwrap_none();
        trace!("New TLS key allocated: {} with dtor {:?}", new_key, dtor);
        new_key
    }

    pub fn delete_tls_key(&mut self, key: TlsKey) -> InterpResult<'tcx> {
        match self.keys.remove(&key) {
            Some(_) => {
                trace!("TLS key {} removed", key);
                Ok(())
            }
            None => throw_unsup!(TlsOutOfBounds),
        }
    }

    pub fn load_tls(
        &mut self,
        key: TlsKey,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx, Scalar<Tag>> {
        match self.keys.get(&key) {
            Some(&TlsEntry { data, .. }) => {
                trace!("TLS key {} loaded: {:?}", key, data);
                Ok(data.unwrap_or_else(|| Scalar::ptr_null(cx).into()))
            }
            None => throw_unsup!(TlsOutOfBounds),
        }
    }

    pub fn store_tls(&mut self, key: TlsKey, new_data: Option<Scalar<Tag>>) -> InterpResult<'tcx> {
        match self.keys.get_mut(&key) {
            Some(&mut TlsEntry { ref mut data, .. }) => {
                trace!("TLS key {} stored: {:?}", key, new_data);
                *data = new_data;
                Ok(())
            }
            None => throw_unsup!(TlsOutOfBounds),
        }
    }

    /// Returns a dtor, its argument and its index, if one is supposed to run
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
        for (&key, &mut TlsEntry { ref mut data, dtor }) in
            thread_local.range_mut((start, Unbounded))
        {
            if let Some(data_scalar) = *data {
                if let Some(dtor) = dtor {
                    let ret = Some((dtor, data_scalar, key));
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
        let mut dtor = this.machine.tls.fetch_tls_dtor(None);
        // FIXME: replace loop by some structure that works with stepping
        while let Some((instance, ptr, key)) = dtor {
            trace!("Running TLS dtor {:?} on {:?}", instance, ptr);
            assert!(!this.is_null(ptr).unwrap(), "Data can't be NULL when dtor is called!");

            let ret_place = MPlaceTy::dangling(this.layout_of(this.tcx.mk_unit())?, this).into();
            this.call_function(
                instance,
                &[ptr.into()],
                Some(ret_place),
                StackPopCleanup::None { cleanup: true },
            )?;

            // step until out of stackframes
            this.run()?;

            dtor = match this.machine.tls.fetch_tls_dtor(Some(key)) {
                dtor @ Some(_) => dtor,
                None => this.machine.tls.fetch_tls_dtor(None),
            };
        }
        // FIXME: On a windows target, call `unsafe extern "system" fn on_tls_callback`.
        Ok(())
    }
}
