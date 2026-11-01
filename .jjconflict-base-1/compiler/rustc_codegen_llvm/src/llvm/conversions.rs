//! Conversions from backend-independent data types to/from LLVM FFI types.

use rustc_codegen_ssa::common::{AtomicRmwBinOp, IntPredicate, RealPredicate, TypeKind};
use rustc_middle::ty::AtomicOrdering;
use rustc_session::config::DebugInfo;
use rustc_target::spec::SymbolVisibility;

use crate::llvm;

/// Helper trait for converting backend-independent types to LLVM-specific
/// types, for FFI purposes.
///
/// FIXME(#147327): These trait/method names were chosen to avoid churn in
/// existing code, but are not great and could probably be made clearer.
pub(crate) trait FromGeneric<T> {
    fn from_generic(other: T) -> Self;
}

/// Helper trait for converting LLVM-specific types to backend-independent
/// types, for FFI purposes.
///
/// FIXME(#147327): These trait/method names were chosen to avoid churn in
/// existing code, but are not great and could probably be made clearer.
pub(crate) trait ToGeneric<T> {
    fn to_generic(&self) -> T;
}

impl FromGeneric<SymbolVisibility> for llvm::Visibility {
    fn from_generic(visibility: SymbolVisibility) -> Self {
        match visibility {
            SymbolVisibility::Hidden => Self::Hidden,
            SymbolVisibility::Protected => Self::Protected,
            SymbolVisibility::Interposable => Self::Default,
        }
    }
}

impl FromGeneric<IntPredicate> for llvm::IntPredicate {
    fn from_generic(int_pred: IntPredicate) -> Self {
        match int_pred {
            IntPredicate::IntEQ => Self::IntEQ,
            IntPredicate::IntNE => Self::IntNE,
            IntPredicate::IntUGT => Self::IntUGT,
            IntPredicate::IntUGE => Self::IntUGE,
            IntPredicate::IntULT => Self::IntULT,
            IntPredicate::IntULE => Self::IntULE,
            IntPredicate::IntSGT => Self::IntSGT,
            IntPredicate::IntSGE => Self::IntSGE,
            IntPredicate::IntSLT => Self::IntSLT,
            IntPredicate::IntSLE => Self::IntSLE,
        }
    }
}

impl FromGeneric<RealPredicate> for llvm::RealPredicate {
    fn from_generic(real_pred: RealPredicate) -> Self {
        match real_pred {
            RealPredicate::RealPredicateFalse => Self::RealPredicateFalse,
            RealPredicate::RealOEQ => Self::RealOEQ,
            RealPredicate::RealOGT => Self::RealOGT,
            RealPredicate::RealOGE => Self::RealOGE,
            RealPredicate::RealOLT => Self::RealOLT,
            RealPredicate::RealOLE => Self::RealOLE,
            RealPredicate::RealONE => Self::RealONE,
            RealPredicate::RealORD => Self::RealORD,
            RealPredicate::RealUNO => Self::RealUNO,
            RealPredicate::RealUEQ => Self::RealUEQ,
            RealPredicate::RealUGT => Self::RealUGT,
            RealPredicate::RealUGE => Self::RealUGE,
            RealPredicate::RealULT => Self::RealULT,
            RealPredicate::RealULE => Self::RealULE,
            RealPredicate::RealUNE => Self::RealUNE,
            RealPredicate::RealPredicateTrue => Self::RealPredicateTrue,
        }
    }
}

impl FromGeneric<AtomicRmwBinOp> for llvm::AtomicRmwBinOp {
    fn from_generic(op: AtomicRmwBinOp) -> Self {
        match op {
            AtomicRmwBinOp::AtomicXchg => Self::AtomicXchg,
            AtomicRmwBinOp::AtomicAdd => Self::AtomicAdd,
            AtomicRmwBinOp::AtomicSub => Self::AtomicSub,
            AtomicRmwBinOp::AtomicAnd => Self::AtomicAnd,
            AtomicRmwBinOp::AtomicNand => Self::AtomicNand,
            AtomicRmwBinOp::AtomicOr => Self::AtomicOr,
            AtomicRmwBinOp::AtomicXor => Self::AtomicXor,
            AtomicRmwBinOp::AtomicMax => Self::AtomicMax,
            AtomicRmwBinOp::AtomicMin => Self::AtomicMin,
            AtomicRmwBinOp::AtomicUMax => Self::AtomicUMax,
            AtomicRmwBinOp::AtomicUMin => Self::AtomicUMin,
        }
    }
}

impl FromGeneric<AtomicOrdering> for llvm::AtomicOrdering {
    fn from_generic(ordering: AtomicOrdering) -> Self {
        match ordering {
            AtomicOrdering::Relaxed => Self::Monotonic,
            AtomicOrdering::Acquire => Self::Acquire,
            AtomicOrdering::Release => Self::Release,
            AtomicOrdering::AcqRel => Self::AcquireRelease,
            AtomicOrdering::SeqCst => Self::SequentiallyConsistent,
        }
    }
}

impl FromGeneric<DebugInfo> for llvm::debuginfo::DebugEmissionKind {
    fn from_generic(kind: DebugInfo) -> Self {
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
        match kind {
            DebugInfo::None => Self::NoDebug,
            DebugInfo::LineDirectivesOnly => Self::DebugDirectivesOnly,
            DebugInfo::LineTablesOnly => Self::LineTablesOnly,
            DebugInfo::Limited | DebugInfo::Full => Self::FullDebug,
        }
    }
}

impl ToGeneric<TypeKind> for llvm::TypeKind {
    fn to_generic(&self) -> TypeKind {
        match self {
            Self::Void => TypeKind::Void,
            Self::Half => TypeKind::Half,
            Self::Float => TypeKind::Float,
            Self::Double => TypeKind::Double,
            Self::X86_FP80 => TypeKind::X86_FP80,
            Self::FP128 => TypeKind::FP128,
            Self::PPC_FP128 => TypeKind::PPC_FP128,
            Self::Label => TypeKind::Label,
            Self::Integer => TypeKind::Integer,
            Self::Function => TypeKind::Function,
            Self::Struct => TypeKind::Struct,
            Self::Array => TypeKind::Array,
            Self::Pointer => TypeKind::Pointer,
            Self::Vector => TypeKind::Vector,
            Self::Metadata => TypeKind::Metadata,
            Self::Token => TypeKind::Token,
            Self::ScalableVector => TypeKind::ScalableVector,
            Self::BFloat => TypeKind::BFloat,
            Self::X86_AMX => TypeKind::X86_AMX,
        }
    }
}
