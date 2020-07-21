use rustc_data_structures::fx::FxHashMap;
use std::collections::hash_map;
use std::slice;

#[derive(Copy, Clone, Debug)]
pub enum CounterOp {
    Add,
    Subtract,
}

pub enum CoverageKind {
    Counter,
    CounterExpression(u32, CounterOp, u32),
}

pub struct CoverageSpan {
    pub start_byte_pos: u32,
    pub end_byte_pos: u32,
}

pub struct CoverageRegion {
    pub kind: CoverageKind,
    pub coverage_span: CoverageSpan,
}

/// Collects all of the coverage regions associated with (a) injected counters, (b) counter
/// expressions (additions or subtraction), and (c) unreachable regions (always counted as zero),
/// for a given Function. Counters and counter expressions are indexed because they can be operands
/// in an expression.
///
/// Note, it's important to distinguish the `unreachable` region type from what LLVM's refers to as
/// a "gap region" (or "gap area"). A gap region is a code region within a counted region (either
/// counter or expression), but the line or lines in the gap region are not executable (such as
/// lines with only whitespace or comments). According to LLVM Code Coverage Mapping documentation,
/// "A count for a gap area is only used as the line execution count if there are no other regions
/// on a line."
#[derive(Default)]
pub struct FunctionCoverageRegions {
    indexed: FxHashMap<u32, CoverageRegion>,
    unreachable: Vec<CoverageSpan>,
}

impl FunctionCoverageRegions {
    pub fn add_counter(&mut self, index: u32, start_byte_pos: u32, end_byte_pos: u32) {
        self.indexed.insert(
            index,
            CoverageRegion {
                kind: CoverageKind::Counter,
                coverage_span: CoverageSpan { start_byte_pos, end_byte_pos },
            },
        );
    }

    pub fn add_counter_expression(
        &mut self,
        index: u32,
        lhs: u32,
        op: CounterOp,
        rhs: u32,
        start_byte_pos: u32,
        end_byte_pos: u32,
    ) {
        self.indexed.insert(
            index,
            CoverageRegion {
                kind: CoverageKind::CounterExpression(lhs, op, rhs),
                coverage_span: CoverageSpan { start_byte_pos, end_byte_pos },
            },
        );
    }

    pub fn add_unreachable(&mut self, start_byte_pos: u32, end_byte_pos: u32) {
        self.unreachable.push(CoverageSpan { start_byte_pos, end_byte_pos });
    }

    pub fn indexed_regions(&self) -> hash_map::Iter<'_, u32, CoverageRegion> {
        self.indexed.iter()
    }

    pub fn unreachable_regions(&self) -> slice::Iter<'_, CoverageSpan> {
        self.unreachable.iter()
    }
}
