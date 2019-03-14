// Not in interpret to make sure we do not use private implementation details

use std::fmt;
use std::error::Error;
use std::borrow::{Borrow, Cow};
use std::hash::Hash;
use std::collections::hash_map::Entry;

use rustc::hir::{self, def_id::DefId};
use rustc::hir::def::Def;
use rustc::mir::interpret::{ConstEvalErr, ErrorHandled};
use rustc::mir;
use rustc::ty::{self, TyCtxt, query::TyCtxtAt};
use rustc::ty::layout::{self, LayoutOf, VariantIdx};
use rustc::ty::subst::Subst;
use rustc::traits::Reveal;
use rustc_data_structures::fx::FxHashMap;
use rustc::util::common::ErrorReported;

use syntax::ast::Mutability;
use syntax::source_map::{Span, DUMMY_SP};

use crate::interpret::{self,
    PlaceTy, MPlaceTy, MemPlace, OpTy, ImmTy, Immediate, Scalar, Pointer,
    RawConst, ConstValue,
    EvalResult, EvalError, EvalErrorKind, GlobalId, EvalContext, StackPopCleanup,
    Allocation, AllocId, MemoryKind,
    snapshot, RefTracking,
};

/// Number of steps until the detector even starts doing anything.
/// Also, a warning is shown to the user when this number is reached.
const STEPS_UNTIL_DETECTOR_ENABLED: isize = 1_000_000;
/// The number of steps between loop detector snapshots.
/// Should be a power of two for performance reasons.
const DETECTOR_SNAPSHOT_PERIOD: isize = 256;

/// The `EvalContext` is only meant to be used to do field and index projections into constants for
/// `simd_shuffle` and const patterns in match arms.
///
/// The function containing the `match` that is currently being analyzed may have generic bounds
/// that inform us about the generic bounds of the constant. E.g., using an associated constant
/// of a function's generic parameter will require knowledge about the bounds on the generic
/// parameter. These bounds are passed to `mk_eval_cx` via the `ParamEnv` argument.
pub(crate) fn mk_eval_cx<'a, 'mir, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    span: Span,
    param_env: ty::ParamEnv<'tcx>,
) -> CompileTimeEvalContext<'a, 'mir, 'tcx> {
    debug!("mk_eval_cx: {:?}", param_env);
    EvalContext::new(tcx.at(span), param_env, CompileTimeInterpreter::new())
}

pub(crate) fn eval_promoted<'a, 'mir, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    cid: GlobalId<'tcx>,
    mir: &'mir mir::Mir<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> EvalResult<'tcx, MPlaceTy<'tcx>> {
    let span = tcx.def_span(cid.instance.def_id());
    let mut ecx = mk_eval_cx(tcx, span, param_env);
    eval_body_using_ecx(&mut ecx, cid, Some(mir), param_env)
}

fn mplace_to_const<'tcx>(
    ecx: &CompileTimeEvalContext<'_, '_, 'tcx>,
    mplace: MPlaceTy<'tcx>,
) -> ty::Const<'tcx> {
    let MemPlace { ptr, align, meta } = *mplace;
    // extract alloc-offset pair
    assert!(meta.is_none());
    let ptr = ptr.to_ptr().unwrap();
    let alloc = ecx.memory.get(ptr.alloc_id).unwrap();
    assert!(alloc.align >= align);
    assert!(alloc.bytes.len() as u64 - ptr.offset.bytes() >= mplace.layout.size.bytes());
    let mut alloc = alloc.clone();
    alloc.align = align;
    // FIXME shouldn't it be the case that `mark_static_initialized` has already
    // interned this?  I thought that is the entire point of that `FinishStatic` stuff?
    let alloc = ecx.tcx.intern_const_alloc(alloc);
    let val = ConstValue::ByRef(ptr, alloc);
    ty::Const { val, ty: mplace.layout.ty }
}

fn op_to_const<'tcx>(
    ecx: &CompileTimeEvalContext<'_, '_, 'tcx>,
    op: OpTy<'tcx>,
) -> ty::Const<'tcx> {
    // We do not normalize just any data.  Only non-union scalars and slices.
    let normalize = match op.layout.abi {
        layout::Abi::Scalar(..) => op.layout.ty.ty_adt_def().map_or(true, |adt| !adt.is_union()),
        layout::Abi::ScalarPair(..) => op.layout.ty.is_slice(),
        _ => false,
    };
    let normalized_op = if normalize {
        Err(*ecx.read_immediate(op).expect("normalization works on validated constants"))
    } else {
        op.try_as_mplace()
    };
    let val = match normalized_op {
        Ok(mplace) => return mplace_to_const(ecx, mplace),
        Err(Immediate::Scalar(x)) =>
            ConstValue::Scalar(x.not_undef().unwrap()),
        Err(Immediate::ScalarPair(a, b)) =>
            ConstValue::Slice(a.not_undef().unwrap(), b.to_usize(ecx).unwrap()),
    };
    ty::Const { val, ty: op.layout.ty }
}

fn eval_body_and_ecx<'a, 'mir, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    cid: GlobalId<'tcx>,
    mir: Option<&'mir mir::Mir<'tcx>>,
    param_env: ty::ParamEnv<'tcx>,
) -> (EvalResult<'tcx, MPlaceTy<'tcx>>, CompileTimeEvalContext<'a, 'mir, 'tcx>) {
    // we start out with the best span we have
    // and try improving it down the road when more information is available
    let span = tcx.def_span(cid.instance.def_id());
    let span = mir.map(|mir| mir.span).unwrap_or(span);
    let mut ecx = EvalContext::new(tcx.at(span), param_env, CompileTimeInterpreter::new());
    let r = eval_body_using_ecx(&mut ecx, cid, mir, param_env);
    (r, ecx)
}

// Returns a pointer to where the result lives
fn eval_body_using_ecx<'mir, 'tcx>(
    ecx: &mut CompileTimeEvalContext<'_, 'mir, 'tcx>,
    cid: GlobalId<'tcx>,
    mir: Option<&'mir mir::Mir<'tcx>>,
    param_env: ty::ParamEnv<'tcx>,
) -> EvalResult<'tcx, MPlaceTy<'tcx>> {
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
    let ret = ecx.allocate(layout, MemoryKind::Stack);

    let name = ty::tls::with(|tcx| tcx.def_path_str(cid.instance.def_id()));
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
    Ok(ret)
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

impl interpret::MayLeak for ! {
    #[inline(always)]
    fn may_leak(self) -> bool {
        // `self` is uninhabited
        self
    }
}

impl<'a, 'mir, 'tcx> interpret::Machine<'a, 'mir, 'tcx>
    for CompileTimeInterpreter<'a, 'mir, 'tcx>
{
    type MemoryKinds = !;
    type PointerTag = ();

    type FrameExtra = ();
    type MemoryExtra = ();
    type AllocExtra = ();

    type MemoryMap = FxHashMap<AllocId, (MemoryKind<!>, Allocation)>;

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
        // Only check non-glue functions
        if let ty::InstanceDef::Item(def_id) = instance.def {
            // Execution might have wandered off into other crates, so we cannot to a stability-
            // sensitive check here.  But we can at least rule out functions that are not const
            // at all.
            if !ecx.tcx.is_const_fn_raw(def_id) {
                // Some functions we support even if they are non-const -- but avoid testing
                // that for const fn!  We certainly do *not* want to actually call the fn
                // though, so be sure we return here.
                return if ecx.hook_fn(instance, args, dest)? {
                    ecx.goto_block(ret)?; // fully evaluated and done
                    Ok(None)
                } else {
                    err!(MachineError(format!("calling non-const function `{}`", instance)))
                };
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
        _left: ImmTy<'tcx>,
        _right: ImmTy<'tcx>,
    ) -> EvalResult<'tcx, (Scalar, bool)> {
        Err(
            ConstEvalError::NeedsRfc("pointer arithmetic or comparison".to_string()).into(),
        )
    }

    fn find_foreign_static(
        _def_id: DefId,
        _tcx: TyCtxtAt<'a, 'tcx, 'tcx>,
        _memory_extra: &(),
    ) -> EvalResult<'tcx, Cow<'tcx, Allocation<Self::PointerTag>>> {
        err!(ReadForeignStatic)
    }

    #[inline(always)]
    fn adjust_static_allocation<'b>(
        alloc: &'b Allocation,
        _memory_extra: &(),
    ) -> Cow<'b, Allocation<Self::PointerTag>> {
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

    #[inline(always)]
    fn tag_new_allocation(
        _ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        ptr: Pointer,
        _kind: MemoryKind<Self::MemoryKinds>,
    ) -> Pointer {
        ptr
    }

    #[inline(always)]
    fn stack_push(
        _ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
    ) -> EvalResult<'tcx> {
        Ok(())
    }

    /// Called immediately before a stack frame gets popped.
    #[inline(always)]
    fn stack_pop(
        _ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        _extra: (),
    ) -> EvalResult<'tcx> {
        Ok(())
    }
}

/// Projects to a field of a (variant of a) const.
// this function uses `unwrap` copiously, because an already validated constant must have valid
// fields and can thus never fail outside of compiler bugs
pub fn const_field<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    variant: Option<VariantIdx>,
    field: mir::Field,
    value: ty::Const<'tcx>,
) -> ty::Const<'tcx> {
    trace!("const_field: {:?}, {:?}", field, value);
    let ecx = mk_eval_cx(tcx, DUMMY_SP, param_env);
    // get the operand again
    let op = ecx.eval_const_to_op(value, None).unwrap();
    // downcast
    let down = match variant {
        None => op,
        Some(variant) => ecx.operand_downcast(op, variant).unwrap(),
    };
    // then project
    let field = ecx.operand_field(down, field.index() as u64).unwrap();
    // and finally move back to the const world, always normalizing because
    // this is not called for statics.
    op_to_const(&ecx, field)
}

// this function uses `unwrap` copiously, because an already validated constant must have valid
// fields and can thus never fail outside of compiler bugs
pub fn const_variant_index<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    val: ty::Const<'tcx>,
) -> VariantIdx {
    trace!("const_variant_index: {:?}", val);
    let ecx = mk_eval_cx(tcx, DUMMY_SP, param_env);
    let op = ecx.eval_const_to_op(val, None).unwrap();
    ecx.read_discriminant(op).unwrap().1
}

pub fn error_to_const_error<'a, 'mir, 'tcx>(
    ecx: &EvalContext<'a, 'mir, 'tcx, CompileTimeInterpreter<'a, 'mir, 'tcx>>,
    mut error: EvalError<'tcx>
) -> ConstEvalErr<'tcx> {
    error.print_backtrace();
    let stacktrace = ecx.generate_stacktrace(None);
    ConstEvalErr { error: error.kind, stacktrace, span: ecx.tcx.span }
}

fn validate_and_turn_into_const<'a, 'tcx>(
    tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    constant: RawConst<'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ::rustc::mir::interpret::ConstEvalResult<'tcx> {
    let cid = key.value;
    let ecx = mk_eval_cx(tcx, tcx.def_span(key.value.instance.def_id()), key.param_env);
    let val = (|| {
        let mplace = ecx.raw_const_to_mplace(constant)?;
        let mut ref_tracking = RefTracking::new(mplace);
        while let Some((mplace, path)) = ref_tracking.todo.pop() {
            ecx.validate_operand(
                mplace.into(),
                path,
                Some(&mut ref_tracking),
                true, // const mode
            )?;
        }
        // Now that we validated, turn this into a proper constant.
        let def_id = cid.instance.def.def_id();
        if tcx.is_static(def_id).is_some() || cid.promoted.is_some() {
            Ok(mplace_to_const(&ecx, mplace))
        } else {
            Ok(op_to_const(&ecx, mplace.into()))
        }
    })();

    val.map_err(|error| {
        let err = error_to_const_error(&ecx, error);
        match err.struct_error(ecx.tcx, "it is undefined behavior to use this value") {
            Ok(mut diag) => {
                diag.note("The rules on what exactly is undefined behavior aren't clear, \
                    so this check might be overzealous. Please open an issue on the rust compiler \
                    repository if you believe it should not be considered undefined behavior",
                );
                diag.emit();
                ErrorHandled::Reported
            }
            Err(err) => err,
        }
    })
}

pub fn const_eval_provider<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ::rustc::mir::interpret::ConstEvalResult<'tcx> {
    // see comment in const_eval_provider for what we're doing here
    if key.param_env.reveal == Reveal::All {
        let mut key = key.clone();
        key.param_env.reveal = Reveal::UserFacing;
        match tcx.const_eval(key) {
            // try again with reveal all as requested
            Err(ErrorHandled::TooGeneric) => {
                // Promoteds should never be "too generic" when getting evaluated.
                // They either don't get evaluated, or we are in a monomorphic context
                assert!(key.value.promoted.is_none());
            },
            // dedupliate calls
            other => return other,
        }
    }
    tcx.const_eval_raw(key).and_then(|val| {
        validate_and_turn_into_const(tcx, val, key)
    })
}

pub fn const_eval_raw_provider<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ::rustc::mir::interpret::ConstEvalRawResult<'tcx> {
    // Because the constant is computed twice (once per value of `Reveal`), we are at risk of
    // reporting the same error twice here. To resolve this, we check whether we can evaluate the
    // constant in the more restrictive `Reveal::UserFacing`, which most likely already was
    // computed. For a large percentage of constants that will already have succeeded. Only
    // associated constants of generic functions will fail due to not enough monomorphization
    // information being available.

    // In case we fail in the `UserFacing` variant, we just do the real computation.
    if key.param_env.reveal == Reveal::All {
        let mut key = key.clone();
        key.param_env.reveal = Reveal::UserFacing;
        match tcx.const_eval_raw(key) {
            // try again with reveal all as requested
            Err(ErrorHandled::TooGeneric) => {},
            // dedupliate calls
            other => return other,
        }
    }
    if cfg!(debug_assertions) {
        // Make sure we format the instance even if we do not print it.
        // This serves as a regression test against an ICE on printing.
        // The next two lines concatenated contain some discussion:
        // https://rust-lang.zulipchat.com/#narrow/stream/146212-t-compiler.2Fconst-eval/
        // subject/anon_const_instance_printing/near/135980032
        let instance = key.value.instance.to_string();
        trace!("const eval: {:?} ({})", key, instance);
    }

    let cid = key.value;
    let def_id = cid.instance.def.def_id();

    if let Some(id) = tcx.hir().as_local_hir_id(def_id) {
        let tables = tcx.typeck_tables_of(def_id);

        // Do match-check before building MIR
        if let Err(ErrorReported) = tcx.check_match(def_id) {
            return Err(ErrorHandled::Reported)
        }

        if let hir::BodyOwnerKind::Const = tcx.hir().body_owner_kind_by_hir_id(id) {
            tcx.mir_const_qualif(def_id);
        }

        // Do not continue into miri if typeck errors occurred; it will fail horribly
        if tables.tainted_by_errors {
            return Err(ErrorHandled::Reported)
        }
    };

    let (res, ecx) = eval_body_and_ecx(tcx, cid, None, key.param_env);
    res.and_then(|place| {
        Ok(RawConst {
            alloc_id: place.to_ptr().expect("we allocated this ptr!").alloc_id,
            ty: place.layout.ty
        })
    }).map_err(|error| {
        let err = error_to_const_error(&ecx, error);
        // errors in statics are always emitted as fatal errors
        if tcx.is_static(def_id).is_some() {
            let reported_err = err.report_as_error(ecx.tcx,
                                                   "could not evaluate static initializer");
            // Ensure that if the above error was either `TooGeneric` or `Reported`
            // an error must be reported.
            if tcx.sess.err_count() == 0 {
                tcx.sess.delay_span_bug(err.span,
                                        &format!("static eval failure did not emit an error: {:#?}",
                                                 reported_err));
            }
            reported_err
        } else if def_id.is_local() {
            // constant defined in this crate, we can figure out a lint level!
            match tcx.describe_def(def_id) {
                // constants never produce a hard error at the definition site. Anything else is
                // a backwards compatibility hazard (and will break old versions of winapi for sure)
                //
                // note that validation may still cause a hard error on this very same constant,
                // because any code that existed before validation could not have failed validation
                // thus preventing such a hard error from being a backwards compatibility hazard
                Some(Def::Const(_)) | Some(Def::AssociatedConst(_)) => {
                    let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
                    err.report_as_lint(
                        tcx.at(tcx.def_span(def_id)),
                        "any use of this value will cause an error",
                        hir_id,
                    )
                },
                // promoting runtime code is only allowed to error if it references broken constants
                // any other kind of error will be reported to the user as a deny-by-default lint
                _ => if let Some(p) = cid.promoted {
                    let span = tcx.optimized_mir(def_id).promoted[p].span;
                    if let EvalErrorKind::ReferencedConstant = err.error {
                        err.report_as_error(
                            tcx.at(span),
                            "evaluation of constant expression failed",
                        )
                    } else {
                        err.report_as_lint(
                            tcx.at(span),
                            "reaching this expression at runtime will panic or abort",
                            tcx.hir().as_local_hir_id(def_id).unwrap(),
                        )
                    }
                // anything else (array lengths, enum initializers, constant patterns) are reported
                // as hard errors
                } else {
                    err.report_as_error(
                        ecx.tcx,
                        "evaluation of constant value failed",
                    )
                },
            }
        } else {
            // use of broken constant from other crate
            err.report_as_error(ecx.tcx, "could not evaluate constant")
        }
    })
}
