use std::collections::BTreeMap;

use rustc::{ty, ty::layout::HasDataLayout, mir};

use super::{EvalResult, EvalErrorKind, Scalar, Evaluator,
            Place, StackPopCleanup, EvalContext};

pub type TlsKey = u128;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TlsEntry<'tcx> {
    pub(crate) data: Scalar, // Will eventually become a map from thread IDs to `Scalar`s, if we ever support more than one thread.
    pub(crate) dtor: Option<ty::Instance<'tcx>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsData<'tcx> {
    /// The Key to use for the next thread-local allocation.
    pub(crate) next_key: TlsKey,

    /// pthreads-style thread-local storage.
    pub(crate) keys: BTreeMap<TlsKey, TlsEntry<'tcx>>,
}

impl<'tcx> Default for TlsData<'tcx> {
    fn default() -> Self {
        TlsData {
            next_key: 1, // start with 1 as we must not use 0 on Windows
            keys: Default::default(),
        }
    }
}

pub trait EvalContextExt<'tcx> {
    fn run_tls_dtors(&mut self) -> EvalResult<'tcx>;
}

impl<'tcx> TlsData<'tcx> {
    pub fn create_tls_key(
        &mut self,
        dtor: Option<ty::Instance<'tcx>>,
        cx: impl HasDataLayout,
    ) -> TlsKey {
        let new_key = self.next_key;
        self.next_key += 1;
        self.keys.insert(
            new_key,
            TlsEntry {
                data: Scalar::ptr_null(cx).into(),
                dtor,
            },
        );
        trace!("New TLS key allocated: {} with dtor {:?}", new_key, dtor);
        new_key
    }

    pub fn delete_tls_key(&mut self, key: TlsKey) -> EvalResult<'tcx> {
        match self.keys.remove(&key) {
            Some(_) => {
                trace!("TLS key {} removed", key);
                Ok(())
            }
            None => err!(TlsOutOfBounds),
        }
    }

    pub fn load_tls(&mut self, key: TlsKey) -> EvalResult<'tcx, Scalar> {
        match self.keys.get(&key) {
            Some(&TlsEntry { data, .. }) => {
                trace!("TLS key {} loaded: {:?}", key, data);
                Ok(data)
            }
            None => err!(TlsOutOfBounds),
        }
    }

    pub fn store_tls(&mut self, key: TlsKey, new_data: Scalar) -> EvalResult<'tcx> {
        match self.keys.get_mut(&key) {
            Some(&mut TlsEntry { ref mut data, .. }) => {
                trace!("TLS key {} stored: {:?}", key, new_data);
                *data = new_data;
                Ok(())
            }
            None => err!(TlsOutOfBounds),
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
        cx: impl HasDataLayout,
    ) -> Option<(ty::Instance<'tcx>, Scalar, TlsKey)> {
        use std::collections::Bound::*;

        let thread_local = &mut self.keys;
        let start = match key {
            Some(key) => Excluded(key),
            None => Unbounded,
        };
        for (&key, &mut TlsEntry { ref mut data, dtor }) in
            thread_local.range_mut((start, Unbounded))
        {
            if !data.is_null() {
                if let Some(dtor) = dtor {
                    let ret = Some((dtor, *data, key));
                    *data = Scalar::ptr_null(cx);
                    return ret;
                }
            }
        }
        None
    }
}

impl<'a, 'mir, 'tcx: 'mir + 'a> EvalContextExt<'tcx> for EvalContext<'a, 'mir, 'tcx, Evaluator<'tcx>> {
    fn run_tls_dtors(&mut self) -> EvalResult<'tcx> {
        let mut dtor = self.machine.tls.fetch_tls_dtor(None, *self.tcx);
        // FIXME: replace loop by some structure that works with stepping
        while let Some((instance, ptr, key)) = dtor {
            trace!("Running TLS dtor {:?} on {:?}", instance, ptr);
            // TODO: Potentially, this has to support all the other possible instances?
            // See eval_fn_call in interpret/terminator/mod.rs
            let mir = self.load_mir(instance.def)?;
            let ret = Place::null(&self);
            self.push_stack_frame(
                instance,
                mir.span,
                mir,
                ret,
                StackPopCleanup::None { cleanup: true },
            )?;
            let arg_local = self.frame().mir.args_iter().next().ok_or_else(
                || EvalErrorKind::AbiViolation("TLS dtor does not take enough arguments.".to_owned()),
            )?;
            let dest = self.eval_place(&mir::Place::Local(arg_local))?;
            self.write_scalar(ptr, dest)?;

            // step until out of stackframes
            self.run()?;

            dtor = match self.machine.tls.fetch_tls_dtor(Some(key), *self.tcx) {
                dtor @ Some(_) => dtor,
                None => self.machine.tls.fetch_tls_dtor(None, *self.tcx),
            };
        }
        // FIXME: On a windows target, call `unsafe extern "system" fn on_tls_callback`.
        Ok(())
    }
}
