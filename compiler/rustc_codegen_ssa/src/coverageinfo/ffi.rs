use rustc_middle::mir::coverage::{CounterValueReference, MappedExpressionIndex};

/// Aligns with [llvm::coverage::Counter::CounterKind](https://github.com/rust-lang/llvm-project/blob/rustc/10.0-2020-05-05/llvm/include/llvm/ProfileData/Coverage/CoverageMapping.h#L91)
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
/// Aligns with [llvm::coverage::Counter](https://github.com/rust-lang/llvm-project/blob/rustc/10.0-2020-05-05/llvm/include/llvm/ProfileData/Coverage/CoverageMapping.h#L98-L99)
/// Important: The Rust struct layout (order and types of fields) must match its C++ counterpart.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Counter {
    // Important: The layout (order and types of fields) must match its C++ counterpart.
    pub kind: CounterKind,
    pub id: u32,
}

impl Counter {
    pub fn zero() -> Self {
        Self { kind: CounterKind::Zero, id: 0 }
    }

    pub fn counter_value_reference(counter_id: CounterValueReference) -> Self {
        Self { kind: CounterKind::CounterValueReference, id: counter_id.into() }
    }

    pub fn expression(mapped_expression_index: MappedExpressionIndex) -> Self {
        Self { kind: CounterKind::Expression, id: mapped_expression_index.into() }
    }
}

/// Aligns with [llvm::coverage::CounterExpression::ExprKind](https://github.com/rust-lang/llvm-project/blob/rustc/10.0-2020-05-05/llvm/include/llvm/ProfileData/Coverage/CoverageMapping.h#L146)
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub enum ExprKind {
    Subtract = 0,
    Add = 1,
}

/// Aligns with [llvm::coverage::CounterExpression](https://github.com/rust-lang/llvm-project/blob/rustc/10.0-2020-05-05/llvm/include/llvm/ProfileData/Coverage/CoverageMapping.h#L147-L148)
/// Important: The Rust struct layout (order and types of fields) must match its C++
/// counterpart.
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
