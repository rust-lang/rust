use rustc_middle::mir::coverage::{CounterId, CovTerm, ExpressionId, SourceRegion};

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
    id: u32,
}

impl Counter {
    /// A `Counter` of kind `Zero`. For this counter kind, the `id` is not used.
    pub(crate) const ZERO: Self = Self { kind: CounterKind::Zero, id: 0 };

    /// Constructs a new `Counter` of kind `CounterValueReference`.
    pub(crate) fn counter_value_reference(counter_id: CounterId) -> Self {
        Self { kind: CounterKind::CounterValueReference, id: counter_id.as_u32() }
    }

    /// Constructs a new `Counter` of kind `Expression`.
    pub(crate) fn expression(expression_id: ExpressionId) -> Self {
        Self { kind: CounterKind::Expression, id: expression_id.as_u32() }
    }

    pub(crate) fn from_term(term: CovTerm) -> Self {
        match term {
            CovTerm::Zero => Self::ZERO,
            CovTerm::Counter(id) => Self::counter_value_reference(id),
            CovTerm::Expression(id) => Self::expression(id),
        }
    }
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

pub(crate) mod mcdc {
    use rustc_middle::mir::coverage::{ConditionId, ConditionInfo, DecisionInfo};

    /// Must match the layout of `LLVMRustMCDCDecisionParameters`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct DecisionParameters {
        bitmap_idx: u32,
        num_conditions: u16,
    }

    type LLVMConditionId = i16;

    /// Must match the layout of `LLVMRustMCDCBranchParameters`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct BranchParameters {
        condition_id: LLVMConditionId,
        condition_ids: [LLVMConditionId; 2],
    }

    impl From<ConditionInfo> for BranchParameters {
        fn from(value: ConditionInfo) -> Self {
            let to_llvm_cond_id = |cond_id: Option<ConditionId>| {
                cond_id.and_then(|id| LLVMConditionId::try_from(id.as_usize()).ok()).unwrap_or(-1)
            };
            let ConditionInfo { condition_id, true_next_id, false_next_id } = value;
            Self {
                condition_id: to_llvm_cond_id(Some(condition_id)),
                condition_ids: [to_llvm_cond_id(false_next_id), to_llvm_cond_id(true_next_id)],
            }
        }
    }

    impl From<DecisionInfo> for DecisionParameters {
        fn from(info: DecisionInfo) -> Self {
            let DecisionInfo { bitmap_idx, num_conditions } = info;
            Self { bitmap_idx, num_conditions }
        }
    }
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
    file_id: u32,

    /// 1-based starting line of the source code span.
    start_line: u32,
    /// 1-based starting column of the source code span.
    start_col: u32,
    /// 1-based ending line of the source code span.
    end_line: u32,
    /// 1-based ending column of the source code span. High bit must be unset.
    end_col: u32,
}

impl CoverageSpan {
    pub(crate) fn from_source_region(file_id: u32, code_region: &SourceRegion) -> Self {
        let &SourceRegion { file_name: _, start_line, start_col, end_line, end_col } = code_region;
        // Internally, LLVM uses the high bit of `end_col` to distinguish between
        // code regions and gap regions, so it can't be used by the column number.
        assert!(end_col & (1u32 << 31) == 0, "high bit of `end_col` must be unset: {end_col:#X}");
        Self { file_id, start_line, start_col, end_line, end_col }
    }
}

/// Must match the layout of `LLVMRustCoverageCodeRegion`.
#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct CodeRegion {
    pub(crate) span: CoverageSpan,
    pub(crate) counter: Counter,
}

/// Must match the layout of `LLVMRustCoverageBranchRegion`.
#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct BranchRegion {
    pub(crate) span: CoverageSpan,
    pub(crate) true_counter: Counter,
    pub(crate) false_counter: Counter,
}

/// Must match the layout of `LLVMRustCoverageMCDCBranchRegion`.
#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct MCDCBranchRegion {
    pub(crate) span: CoverageSpan,
    pub(crate) true_counter: Counter,
    pub(crate) false_counter: Counter,
    pub(crate) mcdc_branch_params: mcdc::BranchParameters,
}

/// Must match the layout of `LLVMRustCoverageMCDCDecisionRegion`.
#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct MCDCDecisionRegion {
    pub(crate) span: CoverageSpan,
    pub(crate) mcdc_decision_params: mcdc::DecisionParameters,
}
