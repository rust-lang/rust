/// Must match the layout of `LLVMRustCounterKind`.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub(crate) enum CounterKind {
    Zero = 0,
    CounterValueReference = 1,
    Expression = 2,
}

/// A reference to an instance of an abstract "counter" that will yield a value in a coverage
/// report. Note that `id` has different interpretations, depending on the `kind`:
///   * For `CounterKind::Zero`, `id` is assumed to be `0`
///   * For `CounterKind::CounterValueReference`,  `id` matches the `counter_id` of the injected
///     instrumentation counter (the `index` argument to the LLVM intrinsic
///     `instrprof.increment()`)
///   * For `CounterKind::Expression`, `id` is the index into the coverage map's array of
///     counter expressions.
///
/// Corresponds to struct `llvm::coverage::Counter`.
///
/// Must match the layout of `LLVMRustCounter`.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub(crate) struct Counter {
    // Important: The layout (order and types of fields) must match its C++ counterpart.
    pub(crate) kind: CounterKind,
    pub(crate) id: u32,
}

impl Counter {
    /// A `Counter` of kind `Zero`. For this counter kind, the `id` is not used.
    pub(crate) const ZERO: Self = Self { kind: CounterKind::Zero, id: 0 };
}

/// Corresponds to enum `llvm::coverage::CounterExpression::ExprKind`.
///
/// Must match the layout of `LLVMRustCounterExprKind`.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub(crate) enum ExprKind {
    Subtract = 0,
    Add = 1,
}

/// Corresponds to struct `llvm::coverage::CounterExpression`.
///
/// Must match the layout of `LLVMRustCounterExpression`.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub(crate) struct CounterExpression {
    pub(crate) kind: ExprKind,
    pub(crate) lhs: Counter,
    pub(crate) rhs: Counter,
}

/// A span of source code coordinates to be embedded in coverage metadata.
///
/// Must match the layout of `LLVMRustCoverageSpan`.
#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct CoverageSpan {
    /// Local index into the function's local-to-global file ID table.
    /// The value at that index is itself an index into the coverage filename
    /// table in the CGU's `__llvm_covmap` section.
    pub(crate) file_id: u32,

    /// 1-based starting line of the source code span.
    pub(crate) start_line: u32,
    /// 1-based starting column of the source code span.
    pub(crate) start_col: u32,
    /// 1-based ending line of the source code span.
    pub(crate) end_line: u32,
    /// 1-based ending column of the source code span. High bit must be unset.
    pub(crate) end_col: u32,
}

/// Must match the layout of `LLVMRustCoverageCodeRegion`.
#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct CodeRegion {
    pub(crate) cov_span: CoverageSpan,
    pub(crate) counter: Counter,
}

/// Must match the layout of `LLVMRustCoverageExpansionRegion`.
#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct ExpansionRegion {
    pub(crate) cov_span: CoverageSpan,
    pub(crate) expanded_file_id: u32,
}

/// Must match the layout of `LLVMRustCoverageBranchRegion`.
#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct BranchRegion {
    pub(crate) cov_span: CoverageSpan,
    pub(crate) true_counter: Counter,
    pub(crate) false_counter: Counter,
}

pub(crate) mod mcdc {
    use rustc_middle::mir::coverage::mcdc;

    use crate::coverageinfo::ffi::{Counter, CoverageSpan};

    /// Must match the layout of `LLVMRustCoverageMCDCDecisionRegion`.
    #[repr(C)]
    #[derive(Debug, Clone)]
    pub(crate) struct DecisionRegion {
        pub(crate) cov_span: CoverageSpan,
        pub(crate) params: DecisionParameters,
    }

    /// Must match the layout of `LLVMRustCoverageMCDCConditionRegion`.
    #[repr(C)]
    #[derive(Debug, Clone)]
    pub(crate) struct ConditionRegion {
        pub(crate) cov_span: CoverageSpan,
        pub(crate) true_counter: Counter,
        pub(crate) false_counter: Counter,
        pub(crate) params: ConditionParameters,
    }

    /// Must match the layout of `LLVMRustCoverageMCDCDecisionParameters`.
    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy)]
    pub(crate) struct DecisionParameters {
        pub(crate) bitmap_idx: u32,
        pub(crate) num_conditions: u16,
    }

    type LLVMConditionID = i16;

    /// Must match the layout of `LLVMRustCoverageMCDCConditionParameters`.
    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy)]
    pub(crate) struct ConditionParameters {
        condition_id: LLVMConditionID,
        condition_ids: [LLVMConditionID; 2],
    }

    impl From<mcdc::ConditionInfo> for ConditionParameters {
        fn from(value: mcdc::ConditionInfo) -> Self {
            let to_llvm_id = |id: Option<mcdc::ConditionId>| {
                id.map(mcdc::ConditionId::as_usize)
                    .and_then(|id| LLVMConditionID::try_from(id).ok())
                    .unwrap_or(-1)
            };
            Self {
                condition_id: to_llvm_id(Some(value.condition_id)),
                condition_ids: [to_llvm_id(value.false_next_id), to_llvm_id(value.true_next_id)],
            }
        }
    }
}
