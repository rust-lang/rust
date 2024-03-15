use rustc_middle::mir::coverage::{CodeRegion, CounterId, CovTerm, ExpressionId, MappingKind};

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
    /// A `Counter` of kind `Zero`. For this counter kind, the `id` is not used.
    pub(crate) const ZERO: Self = Self { kind: CounterKind::Zero, id: 0 };

    /// Constructs a new `Counter` of kind `CounterValueReference`.
    pub fn counter_value_reference(counter_id: CounterId) -> Self {
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

/// Corresponds to enum `llvm::coverage::CounterMappingRegion::RegionKind`.
///
/// Must match the layout of `LLVMRustCounterMappingRegionKind`.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub enum RegionKind {
    /// A CodeRegion associates some code with a counter
    CodeRegion = 0,

    /// An ExpansionRegion represents a file expansion region that associates
    /// a source range with the expansion of a virtual source file, such as
    /// for a macro instantiation or #include file.
    ExpansionRegion = 1,

    /// A SkippedRegion represents a source range with code that was skipped
    /// by a preprocessor or similar means.
    SkippedRegion = 2,

    /// A GapRegion is like a CodeRegion, but its count is only set as the
    /// line execution count when its the only region in the line.
    GapRegion = 3,

    /// A BranchRegion represents leaf-level boolean expressions and is
    /// associated with two counters, each representing the number of times the
    /// expression evaluates to true or false.
    BranchRegion = 4,
}

/// This struct provides LLVM's representation of a "CoverageMappingRegion", encoded into the
/// coverage map, in accordance with the
/// [LLVM Code Coverage Mapping Format](https://github.com/rust-lang/llvm-project/blob/rustc/13.0-2021-09-30/llvm/docs/CoverageMappingFormat.rst#llvm-code-coverage-mapping-format).
/// The struct composes fields representing the `Counter` type and value(s) (injected counter
/// ID, or expression type and operands), the source file (an indirect index into a "filenames
/// array", encoded separately), and source location (start and end positions of the represented
/// code region).
///
/// Corresponds to struct `llvm::coverage::CounterMappingRegion`.
///
/// Must match the layout of `LLVMRustCounterMappingRegion`.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct CounterMappingRegion {
    /// The counter type and type-dependent counter data, if any.
    counter: Counter,

    /// If the `RegionKind` is a `BranchRegion`, this represents the counter
    /// for the false branch of the region.
    false_counter: Counter,

    /// An indirect reference to the source filename. In the LLVM Coverage Mapping Format, the
    /// file_id is an index into a function-specific `virtual_file_mapping` array of indexes
    /// that, in turn, are used to look up the filename for this region.
    file_id: u32,

    /// If the `RegionKind` is an `ExpansionRegion`, the `expanded_file_id` can be used to find
    /// the mapping regions created as a result of macro expansion, by checking if their file id
    /// matches the expanded file id.
    expanded_file_id: u32,

    /// 1-based starting line of the mapping region.
    start_line: u32,

    /// 1-based starting column of the mapping region.
    start_col: u32,

    /// 1-based ending line of the mapping region.
    end_line: u32,

    /// 1-based ending column of the mapping region. If the high bit is set, the current
    /// mapping region is a gap area.
    end_col: u32,

    kind: RegionKind,
}

impl CounterMappingRegion {
    pub(crate) fn from_mapping(
        mapping_kind: &MappingKind,
        local_file_id: u32,
        code_region: &CodeRegion,
    ) -> Self {
        let &CodeRegion { file_name: _, start_line, start_col, end_line, end_col } = code_region;
        match *mapping_kind {
            MappingKind::Code(term) => Self::code_region(
                Counter::from_term(term),
                local_file_id,
                start_line,
                start_col,
                end_line,
                end_col,
            ),
            MappingKind::Branch { true_term, false_term } => Self::branch_region(
                Counter::from_term(true_term),
                Counter::from_term(false_term),
                local_file_id,
                start_line,
                start_col,
                end_line,
                end_col,
            ),
        }
    }

    pub(crate) fn code_region(
        counter: Counter,
        file_id: u32,
        start_line: u32,
        start_col: u32,
        end_line: u32,
        end_col: u32,
    ) -> Self {
        Self {
            counter,
            false_counter: Counter::ZERO,
            file_id,
            expanded_file_id: 0,
            start_line,
            start_col,
            end_line,
            end_col,
            kind: RegionKind::CodeRegion,
        }
    }

    pub(crate) fn branch_region(
        counter: Counter,
        false_counter: Counter,
        file_id: u32,
        start_line: u32,
        start_col: u32,
        end_line: u32,
        end_col: u32,
    ) -> Self {
        Self {
            counter,
            false_counter,
            file_id,
            expanded_file_id: 0,
            start_line,
            start_col,
            end_line,
            end_col,
            kind: RegionKind::BranchRegion,
        }
    }

    // This function might be used in the future; the LLVM API is still evolving, as is coverage
    // support.
    #[allow(dead_code)]
    pub(crate) fn expansion_region(
        file_id: u32,
        expanded_file_id: u32,
        start_line: u32,
        start_col: u32,
        end_line: u32,
        end_col: u32,
    ) -> Self {
        Self {
            counter: Counter::ZERO,
            false_counter: Counter::ZERO,
            file_id,
            expanded_file_id,
            start_line,
            start_col,
            end_line,
            end_col,
            kind: RegionKind::ExpansionRegion,
        }
    }

    // This function might be used in the future; the LLVM API is still evolving, as is coverage
    // support.
    #[allow(dead_code)]
    pub(crate) fn skipped_region(
        file_id: u32,
        start_line: u32,
        start_col: u32,
        end_line: u32,
        end_col: u32,
    ) -> Self {
        Self {
            counter: Counter::ZERO,
            false_counter: Counter::ZERO,
            file_id,
            expanded_file_id: 0,
            start_line,
            start_col,
            end_line,
            end_col,
            kind: RegionKind::SkippedRegion,
        }
    }

    // This function might be used in the future; the LLVM API is still evolving, as is coverage
    // support.
    #[allow(dead_code)]
    pub(crate) fn gap_region(
        counter: Counter,
        file_id: u32,
        start_line: u32,
        start_col: u32,
        end_line: u32,
        end_col: u32,
    ) -> Self {
        Self {
            counter,
            false_counter: Counter::ZERO,
            file_id,
            expanded_file_id: 0,
            start_line,
            start_col,
            end_line,
            end_col: (1_u32 << 31) | end_col,
            kind: RegionKind::GapRegion,
        }
    }
}
