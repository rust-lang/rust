/// Must match the layout of `LLVMRustCounterKind`.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub enum CounterKind {
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
pub struct Counter {
    // Important: The layout (order and types of fields) must match its C++ counterpart.
    pub kind: CounterKind,
    pub id: u32,
}

impl Counter {
    /// A `Counter` of kind `Zero`. For this counter kind, the `id` is not used.
    pub const ZERO: Self = Self { kind: CounterKind::Zero, id: 0 };
}

/// Corresponds to enum `llvm::coverage::CounterExpression::ExprKind`.
///
/// Must match the layout of `LLVMRustCounterExprKind`.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub enum ExprKind {
    Subtract = 0,
    Add = 1,
}

/// Corresponds to struct `llvm::coverage::CounterExpression`.
///
/// Must match the layout of `LLVMRustCounterExpression`.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct CounterExpression {
    pub kind: ExprKind,
    pub lhs: Counter,
    pub rhs: Counter,
}

pub mod mcdc {
    /// Must match the layout of `LLVMRustMCDCDecisionParameters`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct DecisionParameters {
        pub bitmap_idx: u32,
        pub num_conditions: u16,
    }

    pub type LLVMConditionId = i16;

    /// Must match the layout of `LLVMRustMCDCBranchParameters`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct BranchParameters {
        pub condition_id: LLVMConditionId,
        pub condition_ids: [LLVMConditionId; 2],
    }
}

/// A span of source code coordinates to be embedded in coverage metadata.
///
/// Must match the layout of `LLVMRustCoverageSpan`.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct CoverageSpan {
    /// Local index into the function's local-to-global file ID table.
    /// The value at that index is itself an index into the coverage filename
    /// table in the CGU's `__llvm_covmap` section.
    pub file_id: u32,

    /// 1-based starting line of the source code span.
    pub start_line: u32,
    /// 1-based starting column of the source code span.
    pub start_col: u32,
    /// 1-based ending line of the source code span.
    pub end_line: u32,
    /// 1-based ending column of the source code span. High bit must be unset.
    pub end_col: u32,
}

/// Holds tables of the various region types in one struct.
///
/// Don't pass this struct across FFI; pass the individual region tables as
/// pointer/length pairs instead.
///
/// Each field name has a `_regions` suffix for improved readability after
/// exhaustive destructing, which ensures that all region types are handled.
#[derive(Clone, Debug, Default)]
pub struct Regions {
    pub code_regions: Vec<CodeRegion>,
    pub expansion_regions: Vec<ExpansionRegion>,
    pub branch_regions: Vec<BranchRegion>,
    pub mcdc_branch_regions: Vec<MCDCBranchRegion>,
    pub mcdc_decision_regions: Vec<MCDCDecisionRegion>,
}

impl Regions {
    /// Returns true if none of this structure's tables contain any regions.
    pub fn has_no_regions(&self) -> bool {
        let Self {
            code_regions,
            expansion_regions,
            branch_regions,
            mcdc_branch_regions,
            mcdc_decision_regions,
        } = self;

        code_regions.is_empty()
            && expansion_regions.is_empty()
            && branch_regions.is_empty()
            && mcdc_branch_regions.is_empty()
            && mcdc_decision_regions.is_empty()
    }
}

/// Must match the layout of `LLVMRustCoverageCodeRegion`.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct CodeRegion {
    pub cov_span: CoverageSpan,
    pub counter: Counter,
}

/// Must match the layout of `LLVMRustCoverageExpansionRegion`.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct ExpansionRegion {
    pub cov_span: CoverageSpan,
    pub expanded_file_id: u32,
}

/// Must match the layout of `LLVMRustCoverageBranchRegion`.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct BranchRegion {
    pub cov_span: CoverageSpan,
    pub true_counter: Counter,
    pub false_counter: Counter,
}

/// Must match the layout of `LLVMRustCoverageMCDCBranchRegion`.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct MCDCBranchRegion {
    pub cov_span: CoverageSpan,
    pub true_counter: Counter,
    pub false_counter: Counter,
    pub mcdc_branch_params: mcdc::BranchParameters,
}

/// Must match the layout of `LLVMRustCoverageMCDCDecisionRegion`.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct MCDCDecisionRegion {
    pub cov_span: CoverageSpan,
    pub mcdc_decision_params: mcdc::DecisionParameters,
}
