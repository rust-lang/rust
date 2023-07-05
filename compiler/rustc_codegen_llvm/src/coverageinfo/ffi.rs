use rustc_middle::mir::coverage::{CounterValueReference, MappedExpressionIndex};

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
    id: u32,
}

impl Counter {
    /// Constructs a new `Counter` of kind `Zero`. For this `CounterKind`, the
    /// `id` is not used.
    pub fn zero() -> Self {
        Self { kind: CounterKind::Zero, id: 0 }
    }

    /// Constructs a new `Counter` of kind `CounterValueReference`, and converts
    /// the given 1-based counter_id to the required 0-based equivalent for
    /// the `Counter` encoding.
    pub fn counter_value_reference(counter_id: CounterValueReference) -> Self {
        Self { kind: CounterKind::CounterValueReference, id: counter_id.zero_based_index() }
    }

    /// Constructs a new `Counter` of kind `Expression`.
    pub fn expression(mapped_expression_index: MappedExpressionIndex) -> Self {
        Self { kind: CounterKind::Expression, id: mapped_expression_index.into() }
    }

    /// Returns true if the `Counter` kind is `Zero`.
    pub fn is_zero(&self) -> bool {
        matches!(self.kind, CounterKind::Zero)
    }

    /// An explicitly-named function to get the ID value, making it more obvious
    /// that the stored value is now 0-based.
    pub fn zero_based_id(&self) -> u32 {
        debug_assert!(!self.is_zero(), "`id` is undefined for CounterKind::Zero");
        self.id
    }
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

impl CounterExpression {
    pub fn new(lhs: Counter, kind: ExprKind, rhs: Counter) -> Self {
        Self { kind, lhs, rhs }
    }
}
