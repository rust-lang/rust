// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Not in interpret to make sure we do not use private implementation details

use std::fmt;
use std::error::Error;
use std::borrow::{Borrow, Cow};
use std::hash::Hash;
use std::collections::hash_map::Entry;

use rustc::hir::{self, def_id::DefId};
use rustc::mir::interpret::ConstEvalErr;
use rustc::mir;
use rustc::ty::{self, TyCtxt, Instance, query::TyCtxtAt};
use rustc::ty::layout::{self, LayoutOf, TyLayout};
use rustc::ty::subst::Subst;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::fx::FxHashMap;

use syntax::ast::Mutability;
use syntax::source_map::{Span, DUMMY_SP};

use rustc::mir::interpret::{
    EvalResult, EvalError, EvalErrorKind, GlobalId,
    Scalar, Allocation, AllocId, ConstValue,
};
use interpret::{self,
    PlaceTy, MemPlace, OpTy, Operand, Value,
    EvalContext, StackPopCleanup, MemoryKind,
    snapshot,
};

/// Number of steps until the detector even starts doing anything.
/// Also, a warning is shown to the user when this number is reached.
const STEPS_UNTIL_DETECTOR_ENABLED: isize = 1_000_000;
/// The number of steps between loop detector snapshots.
/// Should be a power of two for performance reasons.
const DETECTOR_SNAPSHOT_PERIOD: isize = 256;

pub fn mk_borrowck_eval_cx<'a, 'mir, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: Instance<'tcx>,
    mir: &'mir mir::Mir<'tcx>,
    span: Span,
) -> EvalResult<'tcx, CompileTimeEvalContext<'a, 'mir, 'tcx>> {
    debug!("mk_borrowck_eval_cx: {:?}", instance);
    let param_env = tcx.param_env(instance.def_id());
    let mut ecx = EvalContext::new(tcx.at(span), param_env, CompileTimeInterpreter::new(), ());
    // insert a stack frame so any queries have the correct substs
    // cannot use `push_stack_frame`; if we do `const_prop` explodes
    ecx.stack.push(interpret::Frame {
        block: mir::START_BLOCK,
        locals: IndexVec::new(),
        instance,
        span,
        mir,
        return_place: None,
        return_to_block: StackPopCleanup::Goto(None), // never pop
        stmt: 0,
    });
    Ok(ecx)
}

pub fn mk_eval_cx<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: Instance<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> EvalResult<'tcx, CompileTimeEvalContext<'a, 'tcx, 'tcx>> {
    debug!("mk_eval_cx: {:?}, {:?}", instance, param_env);
    let span = tcx.def_span(instance.def_id());
    let mut ecx = EvalContext::new(tcx.at(span), param_env, CompileTimeInterpreter::new(), ());
    let mir = ecx.load_mir(instance.def)?;
    // insert a stack frame so any queries have the correct substs
    ecx.push_stack_frame(
        instance,
        mir.span,
        mir,
        None,
        StackPopCleanup::Goto(None), // never pop
    )?;
    Ok(ecx)
}

pub(crate) fn eval_promoted<'a, 'mir, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    cid: GlobalId<'tcx>,
    mir: &'mir mir::Mir<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> EvalResult<'tcx, OpTy<'tcx>> {
    let mut ecx = mk_borrowck_eval_cx(tcx, cid.instance, mir, DUMMY_SP).unwrap();
    eval_body_using_ecx(&mut ecx, cid, Some(mir), param_env)
}

pub fn op_to_const<'tcx>(
    ecx: &CompileTimeEvalContext<'_, '_, 'tcx>,
    op: OpTy<'tcx>,
    may_normalize: bool,
) -> EvalResult<'tcx, &'tcx ty::Const<'tcx>> {
    // We do not normalize just any data.  Only scalar layout and fat pointers.
    let normalize = may_normalize
        && match op.layout.abi {
            layout::Abi::Scalar(..) => true,
            layout::Abi::ScalarPair(..) => {
                // Must be a fat pointer
                op.layout.ty.builtin_deref(true).is_some()
            },
            _ => false,
        };
    let normalized_op = if normalize {
        ecx.try_read_value(op)?
    } else {
        match op.op {
            Operand::Indirect(mplace) => Err(mplace),
            Operand::Immediate(val) => Ok(val)
        }
    };
    let val = match normalized_op {
        Err(MemPlace { ptr, align, meta }) => {
            // extract alloc-offset pair
            assert!(meta.is_none());
            let ptr = ptr.to_ptr()?;
            let alloc = ecx.memory.get(ptr.alloc_id)?;
            assert!(alloc.align.abi() >= align.abi());
            assert!(alloc.bytes.len() as u64 - ptr.offset.bytes() >= op.layout.size.bytes());
            let mut alloc = alloc.clone();
            alloc.align = align;
            // FIXME shouldnt it be the case that `mark_static_initialized` has already
            // interned this?  I thought that is the entire point of that `FinishStatic` stuff?
            let alloc = ecx.tcx.intern_const_alloc(alloc);
            ConstValue::ByRef(ptr.alloc_id, alloc, ptr.offset)
        },
        Ok(Value::Scalar(x)) =>
            ConstValue::Scalar(x.not_undef()?),
        Ok(Value::ScalarPair(a, b)) =>
            ConstValue::ScalarPair(a.not_undef()?, b.not_undef()?),
    };
    Ok(ty::Const::from_const_value(ecx.tcx.tcx, val, op.layout.ty))
}

fn eval_body_and_ecx<'a, 'mir, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    cid: GlobalId<'tcx>,
    mir: Option<&'mir mir::Mir<'tcx>>,
    param_env: ty::ParamEnv<'tcx>,
) -> (EvalResult<'tcx, OpTy<'tcx>>, CompileTimeEvalContext<'a, 'mir, 'tcx>) {
    // we start out with the best span we have
    // and try improving it down the road when more information is available
    let span = tcx.def_span(cid.instance.def_id());
    let span = mir.map(|mir| mir.span).unwrap_or(span);
    let mut ecx = EvalContext::new(tcx.at(span), param_env, CompileTimeInterpreter::new(), ());
    let r = eval_body_using_ecx(&mut ecx, cid, mir, param_env);
    (r, ecx)
}

// Returns a pointer to where the result lives
fn eval_body_using_ecx<'mir, 'tcx>(
    ecx: &mut CompileTimeEvalContext<'_, 'mir, 'tcx>,
    cid: GlobalId<'tcx>,
    mir: Option<&'mir mir::Mir<'tcx>>,
    param_env: ty::ParamEnv<'tcx>,
) -> EvalResult<'tcx, OpTy<'tcx>> {
    debug!("eval_body_using_ecx: {:?}, {:?}", cid, param_env);
    let tcx = ecx.tcx.tcx;
    let mut mir = match mir {
        Some(mir) => mir,
        None => ecx.load_mir(cid.instance.def)?,
    };
    if let Some(index) = cid.promoted {
        mir = &mir.promoted[index];
    }
    let layout = ecx.layout_of(mir.return_ty().subst(tcx, cid.instance.substs))?;
    assert!(!layout.is_unsized());
    let ret = ecx.allocate(layout, MemoryKind::Stack)?;

    let name = ty::tls::with(|tcx| tcx.item_path_str(cid.instance.def_id()));
    let prom = cid.promoted.map_or(String::new(), |p| format!("::promoted[{:?}]", p));
    trace!("eval_body_using_ecx: pushing stack frame for global: {}{}", name, prom);
    assert!(mir.arg_count == 0);
    ecx.push_stack_frame(
        cid.instance,
        mir.span,
        mir,
        Some(ret.into()),
        StackPopCleanup::None { cleanup: false },
    )?;

    // The main interpreter loop.
    ecx.run()?;

    // Intern the result
    let internally_mutable = !layout.ty.is_freeze(tcx, param_env, mir.span);
    let is_static = tcx.is_static(cid.instance.def_id());
    let mutability = if is_static == Some(hir::Mutability::MutMutable) || internally_mutable {
        Mutability::Mutable
    } else {
        Mutability::Immutable
    };
    ecx.memory.intern_static(ret.ptr.to_ptr()?.alloc_id, mutability)?;

    debug!("eval_body_using_ecx done: {:?}", *ret);
    Ok(ret.into())
}

impl<'tcx> Into<EvalError<'tcx>> for ConstEvalError {
    fn into(self) -> EvalError<'tcx> {
        EvalErrorKind::MachineError(self.to_string()).into()
    }
}

#[derive(Clone, Debug)]
enum ConstEvalError {
    NeedsRfc(String),
}

impl fmt::Display for ConstEvalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::ConstEvalError::*;
        match *self {
            NeedsRfc(ref msg) => {
                write!(
                    f,
                    "\"{}\" needs an rfc before being allowed inside constants",
                    msg
                )
            }
        }
    }
}

impl Error for ConstEvalError {
    fn description(&self) -> &str {
        use self::ConstEvalError::*;
        match *self {
            NeedsRfc(_) => "this feature needs an rfc before being allowed inside constants",
        }
    }

    fn cause(&self) -> Option<&dyn Error> {
        None
    }
}

// Extra machine state for CTFE, and the Machine instance
pub struct CompileTimeInterpreter<'a, 'mir, 'tcx: 'a+'mir> {
    /// When this value is negative, it indicates the number of interpreter
    /// steps *until* the loop detector is enabled. When it is positive, it is
    /// the number of steps after the detector has been enabled modulo the loop
    /// detector period.
    pub(super) steps_since_detector_enabled: isize,

    /// Extra state to detect loops.
    pub(super) loop_detector: snapshot::InfiniteLoopDetector<'a, 'mir, 'tcx>,
}

impl<'a, 'mir, 'tcx> CompileTimeInterpreter<'a, 'mir, 'tcx> {
    fn new() -> Self {
        CompileTimeInterpreter {
            loop_detector: Default::default(),
            steps_since_detector_enabled: -STEPS_UNTIL_DETECTOR_ENABLED,
        }
    }
}

impl<K: Hash + Eq, V> interpret::AllocMap<K, V> for FxHashMap<K, V> {
    #[inline(always)]
    fn contains_key<Q: ?Sized + Hash + Eq>(&mut self, k: &Q) -> bool
        where K: Borrow<Q>
    {
        FxHashMap::contains_key(self, k)
    }

    #[inline(always)]
    fn insert(&mut self, k: K, v: V) -> Option<V>
    {
        FxHashMap::insert(self, k, v)
    }

    #[inline(always)]
    fn remove<Q: ?Sized + Hash + Eq>(&mut self, k: &Q) -> Option<V>
        where K: Borrow<Q>
    {
        FxHashMap::remove(self, k)
    }

    #[inline(always)]
    fn filter_map_collect<T>(&self, mut f: impl FnMut(&K, &V) -> Option<T>) -> Vec<T> {
        self.iter()
            .filter_map(move |(k, v)| f(k, &*v))
            .collect()
    }

    #[inline(always)]
    fn get_or<E>(
        &self,
        k: K,
        vacant: impl FnOnce() -> Result<V, E>
    ) -> Result<&V, E>
    {
        match self.get(&k) {
            Some(v) => Ok(v),
            None => {
                vacant()?;
                bug!("The CTFE machine shouldn't ever need to extend the alloc_map when reading")
            }
        }
    }

    #[inline(always)]
    fn get_mut_or<E>(
        &mut self,
        k: K,
        vacant: impl FnOnce() -> Result<V, E>
    ) -> Result<&mut V, E>
    {
        match self.entry(k) {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => {
                let v = vacant()?;
                Ok(e.insert(v))
            }
        }
    }
}

type CompileTimeEvalContext<'a, 'mir, 'tcx> =
    EvalContext<'a, 'mir, 'tcx, CompileTimeInterpreter<'a, 'mir, 'tcx>>;

impl<'a, 'mir, 'tcx> interpret::Machine<'a, 'mir, 'tcx>
    for CompileTimeInterpreter<'a, 'mir, 'tcx>
{
    type MemoryData = ();
    type MemoryKinds = !;
    type PointerTag = ();

    type MemoryMap = FxHashMap<AllocId, (MemoryKind<!>, Allocation<()>)>;

    const STATIC_KIND: Option<!> = None; // no copying of statics allowed

    #[inline(always)]
    fn enforce_validity(_ecx: &EvalContext<'a, 'mir, 'tcx, Self>) -> bool {
        false // for now, we don't enforce validity
    }

    fn find_fn(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: Option<PlaceTy<'tcx>>,
        ret: Option<mir::BasicBlock>,
    ) -> EvalResult<'tcx, Option<&'mir mir::Mir<'tcx>>> {
        debug!("eval_fn_call: {:?}", instance);
        if !ecx.tcx.is_const_fn(instance.def_id()) {
            // Some functions we support even if they are non-const -- but avoid testing
            // that for const fn!
            if ecx.hook_fn(instance, args, dest)? {
                ecx.goto_block(ret)?; // fully evaluated and done
                return Ok(None);
            }
        }
        // This is a const fn. Call it.
        Ok(Some(match ecx.load_mir(instance.def) {
            Ok(mir) => mir,
            Err(err) => {
                if let EvalErrorKind::NoMirFor(ref path) = err.kind {
                    return Err(
                        ConstEvalError::NeedsRfc(format!("calling extern function `{}`", path))
                            .into(),
                    );
                }
                return Err(err);
            }
        }))
    }

    fn call_intrinsic(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: PlaceTy<'tcx>,
    ) -> EvalResult<'tcx> {
        if ecx.emulate_intrinsic(instance, args, dest)? {
            return Ok(());
        }
        // An intrinsic that we do not support
        let intrinsic_name = &ecx.tcx.item_name(instance.def_id()).as_str()[..];
        Err(
            ConstEvalError::NeedsRfc(format!("calling intrinsic `{}`", intrinsic_name)).into()
        )
    }

    fn ptr_op(
        _ecx: &EvalContext<'a, 'mir, 'tcx, Self>,
        _bin_op: mir::BinOp,
        _left: Scalar,
        _left_layout: TyLayout<'tcx>,
        _right: Scalar,
        _right_layout: TyLayout<'tcx>,
    ) -> EvalResult<'tcx, (Scalar, bool)> {
        Err(
            ConstEvalError::NeedsRfc("pointer arithmetic or comparison".to_string()).into(),
        )
    }

    fn find_foreign_static(
        _tcx: TyCtxtAt<'a, 'tcx, 'tcx>,
        _def_id: DefId,
    ) -> EvalResult<'tcx, Cow<'tcx, Allocation<Self::PointerTag>>> {
        err!(ReadForeignStatic)
    }

    #[inline(always)]
    fn static_with_default_tag(
        alloc: &'_ Allocation
    ) -> Cow<'_, Allocation<Self::PointerTag>> {
        // We do not use a tag so we can just cheaply forward the reference
        Cow::Borrowed(alloc)
    }

    fn box_alloc(
        _ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        _dest: PlaceTy<'tcx>,
    ) -> EvalResult<'tcx> {
        Err(
            ConstEvalError::NeedsRfc("heap allocations via `box` keyword".to_string()).into(),
        )
    }

    fn before_terminator(ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>) -> EvalResult<'tcx> {
        {
            let steps = &mut ecx.machine.steps_since_detector_enabled;

            *steps += 1;
            if *steps < 0 {
                return Ok(());
            }

            *steps %= DETECTOR_SNAPSHOT_PERIOD;
            if *steps != 0 {
                return Ok(());
            }
        }

        let span = ecx.frame().span;
        ecx.machine.loop_detector.observe_and_analyze(
            &ecx.tcx,
            span,
            &ecx.memory,
            &ecx.stack[..],
        )
    }
}

/// Project to a field of a (variant of a) const
pub fn const_field<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    instance: ty::Instance<'tcx>,
    variant: Option<usize>,
    field: mir::Field,
    value: &'tcx ty::Const<'tcx>,
) -> ::rustc::mir::interpret::ConstEvalResult<'tcx> {
    trace!("const_field: {:?}, {:?}, {:?}", instance, field, value);
    let ecx = mk_eval_cx(tcx, instance, param_env).unwrap();
    let result = (|| {
        // get the operand again
        let op = ecx.const_to_op(value)?;
        // downcast
        let down = match variant {
            None => op,
            Some(variant) => ecx.operand_downcast(op, variant)?
        };
        // then project
        let field = ecx.operand_field(down, field.index() as u64)?;
        // and finally move back to the const world, always normalizing because
        // this is not called for statics.
        op_to_const(&ecx, field, true)
    })();
    result.map_err(|err| {
        let (trace, span) = ecx.generate_stacktrace(None);
        ConstEvalErr {
            error: err,
            stacktrace: trace,
            span,
        }.into()
    })
}

pub fn const_variant_index<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    instance: ty::Instance<'tcx>,
    val: &'tcx ty::Const<'tcx>,
) -> EvalResult<'tcx, usize> {
    trace!("const_variant_index: {:?}, {:?}", instance, val);
    let ecx = mk_eval_cx(tcx, instance, param_env).unwrap();
    let op = ecx.const_to_op(val)?;
    Ok(ecx.read_discriminant(op)?.1)
}

pub fn const_to_allocation_provider<'a, 'tcx>(
    _tcx: TyCtxt<'a, 'tcx, 'tcx>,
    val: &'tcx ty::Const<'tcx>,
) -> &'tcx Allocation {
    // FIXME: This really does not need to be a query.  Instead, we should have a query for statics
    // that returns an allocation directly (or an `AllocId`?), after doing a sanity check of the
    // value and centralizing error reporting.
    match val.val {
        ConstValue::ByRef(_, alloc, offset) => {
            assert_eq!(offset.bytes(), 0);
            return alloc;
        },
        _ => bug!("const_to_allocation called on non-static"),
    }
}

pub fn const_eval_provider<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ::rustc::mir::interpret::ConstEvalResult<'tcx> {
    trace!("const eval: {:?}", key);
    let cid = key.value;
    let def_id = cid.instance.def.def_id();

    if let Some(id) = tcx.hir.as_local_node_id(def_id) {
        let tables = tcx.typeck_tables_of(def_id);
        let span = tcx.def_span(def_id);

        // Do match-check before building MIR
        if tcx.check_match(def_id).is_err() {
            return Err(ConstEvalErr {
                error: EvalErrorKind::CheckMatchError.into(),
                stacktrace: vec![],
                span,
            }.into());
        }

        if let hir::BodyOwnerKind::Const = tcx.hir.body_owner_kind(id) {
            tcx.mir_const_qualif(def_id);
        }

        // Do not continue into miri if typeck errors occurred; it will fail horribly
        if tables.tainted_by_errors {
            return Err(ConstEvalErr {
                error: EvalErrorKind::CheckMatchError.into(),
                stacktrace: vec![],
                span,
            }.into());
        }
    };

    let (res, ecx) = eval_body_and_ecx(tcx, cid, None, key.param_env);
    res.and_then(|op| {
        let normalize = tcx.is_static(def_id).is_none() && cid.promoted.is_none();
        if !normalize {
            // Sanity check: These must always be a MemPlace
            match op.op {
                Operand::Indirect(_) => { /* all is good */ },
                Operand::Immediate(_) => bug!("const eval gave us an Immediate"),
            }
        }
        op_to_const(&ecx, op, normalize)
    }).map_err(|err| {
        let (trace, span) = ecx.generate_stacktrace(None);
        let err = ConstEvalErr {
            error: err,
            stacktrace: trace,
            span,
        };
        if tcx.is_static(def_id).is_some() {
            err.report_as_error(ecx.tcx, "could not evaluate static initializer");
            if tcx.sess.err_count() == 0 {
                span_bug!(span, "static eval failure didn't emit an error: {:#?}", err);
            }
        }
        err.into()
    })
}
