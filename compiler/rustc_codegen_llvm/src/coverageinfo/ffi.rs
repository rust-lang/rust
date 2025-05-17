use rustc_middle::mir::coverage::{CounterId, CovTerm, ExpressionId};

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

/// Holds tables of the various region types in one struct.
///
/// Don't pass this struct across FFI; pass the individual region tables as
/// pointer/length pairs instead.
///
/// Each field name has a `_regions` suffix for improved readability after
/// exhaustive destructing, which ensures that all region types are handled.
#[derive(Clone, Debug, Default)]
pub(crate) struct Regions {
    pub(crate) code_regions: Vec<CodeRegion>,
    pub(crate) expansion_regions: Vec<ExpansionRegion>,
    pub(crate) branch_regions: Vec<BranchRegion>,
    pub(crate) mcdc_branch_regions: Vec<MCDCBranchRegion>,
    pub(crate) mcdc_decision_regions: Vec<MCDCDecisionRegion>,
}

impl Regions {
    /// Returns true if none of this structure's tables contain any regions.
    pub(crate) fn has_no_regions(&self) -> bool {
        // Every region has a span, so if there are no spans then there are no regions.
        self.all_cov_spans().next().is_none()
    }

    pub(crate) fn all_cov_spans(&self) -> impl Iterator<Item = &CoverageSpan> {
        macro_rules! iter_cov_spans {
            ( $( $regions:expr ),* $(,)? ) => {
                std::iter::empty()
                $(
                    .chain( $regions.iter().map(|region| &region.cov_span) )
                )*
            }
        }

        let Self {
            code_regions,
            expansion_regions,
            branch_regions,
            mcdc_branch_regions,
            mcdc_decision_regions,
        } = self;

        iter_cov_spans!(
            code_regions,
            expansion_regions,
            branch_regions,
            mcdc_branch_regions,
            mcdc_decision_regions,
        )
    }
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

/// Must match the layout of `LLVMRustCoverageMCDCBranchRegion`.
#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct MCDCBranchRegion {
    pub(crate) cov_span: CoverageSpan,
    pub(crate) true_counter: Counter,
    pub(crate) false_counter: Counter,
    pub(crate) mcdc_branch_params: mcdc::BranchParameters,
}

/// Must match the layout of `LLVMRustCoverageMCDCDecisionRegion`.
#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct MCDCDecisionRegion {
    pub(crate) cov_span: CoverageSpan,
    pub(crate) mcdc_decision_params: mcdc::DecisionParameters,
}
