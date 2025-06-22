//! Simple conversions between generic and LLVM-specific types.
use rustc_llvm::ffi;

pub(crate) fn int_predicate_from_generic(
    intpre: rustc_codegen_ssa::common::IntPredicate,
) -> ffi::IntPredicate {
    use rustc_codegen_ssa::common::IntPredicate as Common;
    match intpre {
        Common::IntEQ => ffi::IntPredicate::IntEQ,
        Common::IntNE => ffi::IntPredicate::IntNE,
        Common::IntUGT => ffi::IntPredicate::IntUGT,
        Common::IntUGE => ffi::IntPredicate::IntUGE,
        Common::IntULT => ffi::IntPredicate::IntULT,
        Common::IntULE => ffi::IntPredicate::IntULE,
        Common::IntSGT => ffi::IntPredicate::IntSGT,
        Common::IntSGE => ffi::IntPredicate::IntSGE,
        Common::IntSLT => ffi::IntPredicate::IntSLT,
        Common::IntSLE => ffi::IntPredicate::IntSLE,
    }
}

pub(crate) fn real_predicate_from_generic(
    realp: rustc_codegen_ssa::common::RealPredicate,
) -> ffi::RealPredicate {
    use rustc_codegen_ssa::common::RealPredicate as Common;
    match realp {
        Common::RealPredicateFalse => ffi::RealPredicate::RealPredicateFalse,
        Common::RealOEQ => ffi::RealPredicate::RealOEQ,
        Common::RealOGT => ffi::RealPredicate::RealOGT,
        Common::RealOGE => ffi::RealPredicate::RealOGE,
        Common::RealOLT => ffi::RealPredicate::RealOLT,
        Common::RealOLE => ffi::RealPredicate::RealOLE,
        Common::RealONE => ffi::RealPredicate::RealONE,
        Common::RealORD => ffi::RealPredicate::RealORD,
        Common::RealUNO => ffi::RealPredicate::RealUNO,
        Common::RealUEQ => ffi::RealPredicate::RealUEQ,
        Common::RealUGT => ffi::RealPredicate::RealUGT,
        Common::RealUGE => ffi::RealPredicate::RealUGE,
        Common::RealULT => ffi::RealPredicate::RealULT,
        Common::RealULE => ffi::RealPredicate::RealULE,
        Common::RealUNE => ffi::RealPredicate::RealUNE,
        Common::RealPredicateTrue => ffi::RealPredicate::RealPredicateTrue,
    }
}

pub(crate) fn type_kind_to_generic(
    type_kind: ffi::TypeKind,
) -> rustc_codegen_ssa::common::TypeKind {
    use rustc_codegen_ssa::common::TypeKind as Common;
    match type_kind {
        ffi::TypeKind::Void => Common::Void,
        ffi::TypeKind::Half => Common::Half,
        ffi::TypeKind::Float => Common::Float,
        ffi::TypeKind::Double => Common::Double,
        ffi::TypeKind::X86_FP80 => Common::X86_FP80,
        ffi::TypeKind::FP128 => Common::FP128,
        ffi::TypeKind::PPC_FP128 => Common::PPC_FP128,
        ffi::TypeKind::Label => Common::Label,
        ffi::TypeKind::Integer => Common::Integer,
        ffi::TypeKind::Function => Common::Function,
        ffi::TypeKind::Struct => Common::Struct,
        ffi::TypeKind::Array => Common::Array,
        ffi::TypeKind::Pointer => Common::Pointer,
        ffi::TypeKind::Vector => Common::Vector,
        ffi::TypeKind::Metadata => Common::Metadata,
        ffi::TypeKind::Token => Common::Token,
        ffi::TypeKind::ScalableVector => Common::ScalableVector,
        ffi::TypeKind::BFloat => Common::BFloat,
        ffi::TypeKind::X86_AMX => Common::X86_AMX,
    }
}

pub(crate) fn visibility_from_generic(
    visibility: rustc_target::spec::SymbolVisibility,
) -> ffi::Visibility {
    use rustc_target::spec::SymbolVisibility;
    match visibility {
        SymbolVisibility::Hidden => ffi::Visibility::Hidden,
        SymbolVisibility::Protected => ffi::Visibility::Protected,
        SymbolVisibility::Interposable => ffi::Visibility::Default,
    }
}

pub(crate) fn atomic_rmw_bin_op_from_generic(
    op: rustc_codegen_ssa::common::AtomicRmwBinOp,
) -> ffi::AtomicRmwBinOp {
    use rustc_codegen_ssa::common::AtomicRmwBinOp as Common;
    match op {
        Common::AtomicXchg => ffi::AtomicRmwBinOp::AtomicXchg,
        Common::AtomicAdd => ffi::AtomicRmwBinOp::AtomicAdd,
        Common::AtomicSub => ffi::AtomicRmwBinOp::AtomicSub,
        Common::AtomicAnd => ffi::AtomicRmwBinOp::AtomicAnd,
        Common::AtomicNand => ffi::AtomicRmwBinOp::AtomicNand,
        Common::AtomicOr => ffi::AtomicRmwBinOp::AtomicOr,
        Common::AtomicXor => ffi::AtomicRmwBinOp::AtomicXor,
        Common::AtomicMax => ffi::AtomicRmwBinOp::AtomicMax,
        Common::AtomicMin => ffi::AtomicRmwBinOp::AtomicMin,
        Common::AtomicUMax => ffi::AtomicRmwBinOp::AtomicUMax,
        Common::AtomicUMin => ffi::AtomicRmwBinOp::AtomicUMin,
    }
}

pub(crate) fn atomic_ordering_from_generic(
    ao: rustc_middle::ty::AtomicOrdering,
) -> ffi::AtomicOrdering {
    use rustc_middle::ty::AtomicOrdering as Common;
    match ao {
        Common::Relaxed => ffi::AtomicOrdering::Monotonic,
        Common::Acquire => ffi::AtomicOrdering::Acquire,
        Common::Release => ffi::AtomicOrdering::Release,
        Common::AcqRel => ffi::AtomicOrdering::AcquireRelease,
        Common::SeqCst => ffi::AtomicOrdering::SequentiallyConsistent,
    }
}

pub(crate) fn debug_emission_kind_from_generic(
    kind: rustc_session::config::DebugInfo,
) -> ffi::debuginfo::DebugEmissionKind {
    // We should be setting LLVM's emission kind to `LineTablesOnly` if
    // we are compiling with "limited" debuginfo. However, some of the
    // existing tools relied on slightly more debuginfo being generated than
    // would be the case with `LineTablesOnly`, and we did not want to break
    // these tools in a "drive-by fix", without a good idea or plan about
    // what limited debuginfo should exactly look like. So for now we are
    // instead adding a new debuginfo option "line-tables-only" so as to
    // not break anything and to allow users to have 'limited' debug info.
    //
    // See https://github.com/rust-lang/rust/issues/60020 for details.
    use rustc_session::config::DebugInfo;
    match kind {
        DebugInfo::None => ffi::debuginfo::DebugEmissionKind::NoDebug,
        DebugInfo::LineDirectivesOnly => ffi::debuginfo::DebugEmissionKind::DebugDirectivesOnly,
        DebugInfo::LineTablesOnly => ffi::debuginfo::DebugEmissionKind::LineTablesOnly,
        DebugInfo::Limited | DebugInfo::Full => ffi::debuginfo::DebugEmissionKind::FullDebug,
    }
}

/// Constructs a new `Counter` of kind `CounterValueReference`.
pub(crate) fn counter_from_counter_value_reference(
    counter_id: rustc_middle::mir::coverage::CounterId,
) -> ffi::Counter {
    ffi::Counter { kind: ffi::CounterKind::CounterValueReference, id: counter_id.as_u32() }
}

/// Constructs a new `Counter` of kind `Expression`.
pub(crate) fn counter_from_expression(
    expression_id: rustc_middle::mir::coverage::ExpressionId,
) -> ffi::Counter {
    ffi::Counter { kind: ffi::CounterKind::Expression, id: expression_id.as_u32() }
}

pub(crate) fn counter_from_term(term: rustc_middle::mir::coverage::CovTerm) -> ffi::Counter {
    use rustc_middle::mir::coverage::CovTerm;
    match term {
        CovTerm::Zero => ffi::Counter::ZERO,
        CovTerm::Counter(id) => counter_from_counter_value_reference(id),
        CovTerm::Expression(id) => counter_from_expression(id),
    }
}

pub(crate) fn branch_parameters_from_condition_info(
    value: rustc_middle::mir::coverage::ConditionInfo,
) -> ffi::mcdc::BranchParameters {
    let to_llvm_cond_id = |cond_id: Option<rustc_middle::mir::coverage::ConditionId>| {
        cond_id
            .and_then(|id| ffi::mcdc::LLVMConditionId::try_from(id.as_usize()).ok())
            .unwrap_or(-1)
    };
    let rustc_middle::mir::coverage::ConditionInfo { condition_id, true_next_id, false_next_id } =
        value;
    ffi::mcdc::BranchParameters {
        condition_id: to_llvm_cond_id(Some(condition_id)),
        condition_ids: [to_llvm_cond_id(false_next_id), to_llvm_cond_id(true_next_id)],
    }
}

pub(crate) fn decision_parameters_from_decision_info(
    info: rustc_middle::mir::coverage::DecisionInfo,
) -> ffi::mcdc::DecisionParameters {
    let rustc_middle::mir::coverage::DecisionInfo { bitmap_idx, num_conditions } = info;
    ffi::mcdc::DecisionParameters { bitmap_idx, num_conditions }
}
