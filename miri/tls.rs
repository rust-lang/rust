use rustc::{ty, mir};

use super::{TlsKey, TlsEntry, EvalResult, EvalErrorKind, Pointer, Memory, Evaluator, Place,
            StackPopCleanup, EvalContext};

pub trait MemoryExt<'tcx> {
    fn create_tls_key(&mut self, dtor: Option<ty::Instance<'tcx>>) -> TlsKey;
    fn delete_tls_key(&mut self, key: TlsKey) -> EvalResult<'tcx>;
    fn load_tls(&mut self, key: TlsKey) -> EvalResult<'tcx, Pointer>;
    fn store_tls(&mut self, key: TlsKey, new_data: Pointer) -> EvalResult<'tcx>;
    fn fetch_tls_dtor(
        &mut self,
        key: Option<TlsKey>,
    ) -> EvalResult<'tcx, Option<(ty::Instance<'tcx>, Pointer, TlsKey)>>;
}

pub trait EvalContextExt<'tcx> {
    fn run_tls_dtors(&mut self) -> EvalResult<'tcx>;
}

impl<'a, 'tcx: 'a> MemoryExt<'tcx> for Memory<'a, 'tcx, Evaluator<'tcx>> {
    fn create_tls_key(&mut self, dtor: Option<ty::Instance<'tcx>>) -> TlsKey {
        let new_key = self.data.next_thread_local;
        self.data.next_thread_local += 1;
        self.data.thread_local.insert(
            new_key,
            TlsEntry {
                data: Pointer::null(),
                dtor,
            },
        );
        trace!("New TLS key allocated: {} with dtor {:?}", new_key, dtor);
        return new_key;
    }

    fn delete_tls_key(&mut self, key: TlsKey) -> EvalResult<'tcx> {
        return match self.data.thread_local.remove(&key) {
            Some(_) => {
                trace!("TLS key {} removed", key);
                Ok(())
            }
            None => err!(TlsOutOfBounds),
        };
    }

    fn load_tls(&mut self, key: TlsKey) -> EvalResult<'tcx, Pointer> {
        return match self.data.thread_local.get(&key) {
            Some(&TlsEntry { data, .. }) => {
                trace!("TLS key {} loaded: {:?}", key, data);
                Ok(data)
            }
            None => err!(TlsOutOfBounds),
        };
    }

    fn store_tls(&mut self, key: TlsKey, new_data: Pointer) -> EvalResult<'tcx> {
        return match self.data.thread_local.get_mut(&key) {
            Some(&mut TlsEntry { ref mut data, .. }) => {
                trace!("TLS key {} stored: {:?}", key, new_data);
                *data = new_data;
                Ok(())
            }
            None => err!(TlsOutOfBounds),
        };
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
    ) -> EvalResult<'tcx, Option<(ty::Instance<'tcx>, Pointer, TlsKey)>> {
        use std::collections::Bound::*;
        let start = match key {
            Some(key) => Excluded(key),
            None => Unbounded,
        };
        for (&key, &mut TlsEntry { ref mut data, dtor }) in
            self.data.thread_local.range_mut((start, Unbounded))
        {
            if !data.is_null()? {
                if let Some(dtor) = dtor {
                    let ret = Some((dtor, *data, key));
                    *data = Pointer::null();
                    return Ok(ret);
                }
            }
        }
        return Ok(None);
    }
}

impl<'a, 'tcx: 'a> EvalContextExt<'tcx> for EvalContext<'a, 'tcx, Evaluator<'tcx>> {
    fn run_tls_dtors(&mut self) -> EvalResult<'tcx> {
        let mut dtor = self.memory.fetch_tls_dtor(None)?;
        // FIXME: replace loop by some structure that works with stepping
        while let Some((instance, ptr, key)) = dtor {
            trace!("Running TLS dtor {:?} on {:?}", instance, ptr);
            // TODO: Potentially, this has to support all the other possible instances?
            // See eval_fn_call in interpret/terminator/mod.rs
            let mir = self.load_mir(instance.def)?;
            self.push_stack_frame(
                instance,
                mir.span,
                mir,
                Place::undef(),
                StackPopCleanup::None,
            )?;
            let arg_local = self.frame().mir.args_iter().next().ok_or(
                EvalErrorKind::AbiViolation("TLS dtor does not take enough arguments.".to_owned()),
            )?;
            let dest = self.eval_place(&mir::Place::Local(arg_local))?;
            let ty = self.tcx.mk_mut_ptr(self.tcx.types.u8);
            self.write_ptr(dest, ptr, ty)?;

            // step until out of stackframes
            while self.step()? {}

            dtor = match self.memory.fetch_tls_dtor(Some(key))? {
                dtor @ Some(_) => dtor,
                None => self.memory.fetch_tls_dtor(None)?,
            };
        }
        Ok(())
    }
}
