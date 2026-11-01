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
