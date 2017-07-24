#![feature(
    i128_type,
    rustc_private,
)]

// From rustc.
#[macro_use]
extern crate log;
extern crate log_settings;
extern crate rustc;
extern crate rustc_const_math;
extern crate rustc_data_structures;
extern crate syntax;

use rustc::ty::{self, TyCtxt};
use rustc::hir::def_id::DefId;
use rustc::mir;

use std::collections::{
    HashMap,
    BTreeMap,
};

extern crate rustc_miri;
pub use rustc_miri::interpret::*;

mod missing_fns;

use missing_fns::EvalContextExt as MissingFnsEvalContextExt;

pub fn eval_main<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    main_id: DefId,
    start_wrapper: Option<DefId>,
    limits: ResourceLimits,
) {
    fn run_main<'a, 'tcx: 'a>(
        ecx: &mut rustc_miri::interpret::EvalContext<'a, 'tcx, Evaluator>,
        main_id: DefId,
        start_wrapper: Option<DefId>,
    ) -> EvalResult<'tcx> {
        let main_instance = ty::Instance::mono(ecx.tcx, main_id);
        let main_mir = ecx.load_mir(main_instance.def)?;
        let mut cleanup_ptr = None; // Pointer to be deallocated when we are done

        if !main_mir.return_ty.is_nil() || main_mir.arg_count != 0 {
            return Err(EvalError::Unimplemented("miri does not support main functions without `fn()` type signatures".to_owned()));
        }

        if let Some(start_id) = start_wrapper {
            let start_instance = ty::Instance::mono(ecx.tcx, start_id);
            let start_mir = ecx.load_mir(start_instance.def)?;

            if start_mir.arg_count != 3 {
                return Err(EvalError::AbiViolation(format!("'start' lang item should have three arguments, but has {}", start_mir.arg_count)));
            }

            // Return value
            let size = ecx.tcx.data_layout.pointer_size.bytes();
            let align = ecx.tcx.data_layout.pointer_align.abi();
            let ret_ptr = ecx.memory_mut().allocate(size, align, Kind::Stack)?;
            cleanup_ptr = Some(ret_ptr);

            // Push our stack frame
            ecx.push_stack_frame(
                start_instance,
                start_mir.span,
                start_mir,
                Lvalue::from_ptr(ret_ptr),
                StackPopCleanup::None,
            )?;

            let mut args = ecx.frame().mir.args_iter();

            // First argument: pointer to main()
            let main_ptr = ecx.memory_mut().create_fn_alloc(main_instance);
            let dest = ecx.eval_lvalue(&mir::Lvalue::Local(args.next().unwrap()))?;
            let main_ty = main_instance.def.def_ty(ecx.tcx);
            let main_ptr_ty = ecx.tcx.mk_fn_ptr(main_ty.fn_sig(ecx.tcx));
            ecx.write_value(Value::ByVal(PrimVal::Ptr(main_ptr)), dest, main_ptr_ty)?;

            // Second argument (argc): 0
            let dest = ecx.eval_lvalue(&mir::Lvalue::Local(args.next().unwrap()))?;
            let ty = ecx.tcx.types.isize;
            ecx.write_null(dest, ty)?;

            // Third argument (argv): 0
            let dest = ecx.eval_lvalue(&mir::Lvalue::Local(args.next().unwrap()))?;
            let ty = ecx.tcx.mk_imm_ptr(ecx.tcx.mk_imm_ptr(ecx.tcx.types.u8));
            ecx.write_null(dest, ty)?;
        } else {
            ecx.push_stack_frame(
                main_instance,
                main_mir.span,
                main_mir,
                Lvalue::undef(),
                StackPopCleanup::None,
            )?;
        }

        while ecx.step()? {}
        ecx.finish()?;
        if let Some(cleanup_ptr) = cleanup_ptr {
            ecx.memory_mut().deallocate(cleanup_ptr, None, Kind::Stack)?;
        }
        Ok(())
    }

    let mut ecx = EvalContext::new(tcx, limits, Default::default(), Default::default());
    match run_main(&mut ecx, main_id, start_wrapper) {
        Ok(()) => {
            let leaks = ecx.memory().leak_report();
            if leaks != 0 {
                tcx.sess.err("the evaluated program leaked memory");
            }
        }
        Err(e) => {
            ecx.report(&e);
        }
    }
}

struct Evaluator;
#[derive(Default)]
struct EvaluatorData {
    /// Environment variables set by `setenv`
    /// Miri does not expose env vars from the host to the emulated program
    pub(crate) env_vars: HashMap<Vec<u8>, MemoryPointer>,
}

pub type TlsKey = usize;

#[derive(Copy, Clone, Debug)]
pub struct TlsEntry<'tcx> {
    data: Pointer, // Will eventually become a map from thread IDs to `Pointer`s, if we ever support more than one thread.
    dtor: Option<ty::Instance<'tcx>>,
}

#[derive(Default)]
struct MemoryData<'tcx> {
    /// The Key to use for the next thread-local allocation.
    next_thread_local: TlsKey,

    /// pthreads-style thread-local storage.
    thread_local: BTreeMap<TlsKey, TlsEntry<'tcx>>,
}

trait EvalContextExt<'tcx> {
    fn finish(&mut self) -> EvalResult<'tcx>;
}

impl<'a, 'tcx> EvalContextExt<'tcx> for EvalContext<'a, 'tcx, Evaluator> {
    fn finish(&mut self) -> EvalResult<'tcx> {
        let mut dtor = self.memory.fetch_tls_dtor(None)?;
        // FIXME: replace loop by some structure that works with stepping
        while let Some((instance, ptr, key)) = dtor {
            trace!("Running TLS dtor {:?} on {:?}", instance, ptr);
            // TODO: Potentially, this has to support all the other possible instances? See eval_fn_call in terminator/mod.rs
            let mir = self.load_mir(instance.def)?;
            self.push_stack_frame(
                instance,
                mir.span,
                mir,
                Lvalue::undef(),
                StackPopCleanup::None,
            )?;
            let arg_local = self.frame().mir.args_iter().next().ok_or(EvalError::AbiViolation("TLS dtor does not take enough arguments.".to_owned()))?;
            let dest = self.eval_lvalue(&mir::Lvalue::Local(arg_local))?;
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

trait MemoryExt<'tcx> {
    fn create_tls_key(&mut self, dtor: Option<ty::Instance<'tcx>>) -> TlsKey;
    fn delete_tls_key(&mut self, key: TlsKey) -> EvalResult<'tcx>;
    fn load_tls(&mut self, key: TlsKey) -> EvalResult<'tcx, Pointer>;
    fn store_tls(&mut self, key: TlsKey, new_data: Pointer) -> EvalResult<'tcx>;
    fn fetch_tls_dtor(&mut self, key: Option<TlsKey>) -> EvalResult<'tcx, Option<(ty::Instance<'tcx>, Pointer, TlsKey)>>;
}

impl<'a, 'tcx: 'a> MemoryExt<'tcx> for Memory<'a, 'tcx, Evaluator> {
    fn create_tls_key(&mut self, dtor: Option<ty::Instance<'tcx>>) -> TlsKey {
        let new_key = self.data.next_thread_local;
        self.data.next_thread_local += 1;
        self.data.thread_local.insert(new_key, TlsEntry { data: Pointer::null(), dtor });
        trace!("New TLS key allocated: {} with dtor {:?}", new_key, dtor);
        return new_key;
    }

    fn delete_tls_key(&mut self, key: TlsKey) -> EvalResult<'tcx> {
        return match self.data.thread_local.remove(&key) {
            Some(_) => {
                trace!("TLS key {} removed", key);
                Ok(())
            },
            None => Err(EvalError::TlsOutOfBounds)
        }
    }

    fn load_tls(&mut self, key: TlsKey) -> EvalResult<'tcx, Pointer> {
        return match self.data.thread_local.get(&key) {
            Some(&TlsEntry { data, .. }) => {
                trace!("TLS key {} loaded: {:?}", key, data);
                Ok(data)
            },
            None => Err(EvalError::TlsOutOfBounds)
        }
    }

    fn store_tls(&mut self, key: TlsKey, new_data: Pointer) -> EvalResult<'tcx> {
        return match self.data.thread_local.get_mut(&key) {
            Some(&mut TlsEntry { ref mut data, .. }) => {
                trace!("TLS key {} stored: {:?}", key, new_data);
                *data = new_data;
                Ok(())
            },
            None => Err(EvalError::TlsOutOfBounds)
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
    fn fetch_tls_dtor(&mut self, key: Option<TlsKey>) -> EvalResult<'tcx, Option<(ty::Instance<'tcx>, Pointer, TlsKey)>> {
        use std::collections::Bound::*;
        let start = match key {
            Some(key) => Excluded(key),
            None => Unbounded,
        };
        for (&key, &mut TlsEntry { ref mut data, dtor }) in self.data.thread_local.range_mut((start, Unbounded)) {
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

impl<'tcx> Machine<'tcx> for Evaluator {
    type Data = EvaluatorData;
    type MemoryData = MemoryData<'tcx>;
    /// Returns Ok() when the function was handled, fail otherwise
    fn call_missing_fn<'a>(
        ecx: &mut EvalContext<'a, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        destination: Option<(Lvalue<'tcx>, mir::BasicBlock)>,
        arg_operands: &[mir::Operand<'tcx>],
        sig: ty::FnSig<'tcx>,
        path: String,
    ) -> EvalResult<'tcx> {
        ecx.call_missing_fn(instance, destination, arg_operands, sig, path)
    }
}
