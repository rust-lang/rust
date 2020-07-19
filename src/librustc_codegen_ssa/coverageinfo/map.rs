use rustc_data_structures::sync::Lrc;
use rustc_middle::mir;
use rustc_span::source_map::{Pos, SourceFile, SourceMap};
use rustc_span::{BytePos, FileName, RealFileName};

use std::cmp::{Ord, Ordering};
use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub enum CounterOp {
    // Note the order (and therefore the default values) is important. With the attribute
    // `#[repr(C)]`, this enum matches the layout of the LLVM enum defined for the nested enum,
    // `llvm::coverage::CounterExpression::ExprKind`, as shown in the following source snippet:
    // https://github.com/rust-lang/llvm-project/blob/f208b70fbc4dee78067b3c5bd6cb92aa3ba58a1e/llvm/include/llvm/ProfileData/Coverage/CoverageMapping.h#L146
    Subtract,
    Add,
}

#[derive(Copy, Clone, Debug)]
pub enum CoverageKind {
    Counter,
    CounterExpression(u32, CounterOp, u32),
    Unreachable,
}

#[derive(Clone, Debug)]
pub struct CoverageRegion {
    pub kind: CoverageKind,
    pub start_byte_pos: u32,
    pub end_byte_pos: u32,
}

impl CoverageRegion {
    pub fn source_loc(&self, source_map: &SourceMap) -> Option<(Lrc<SourceFile>, CoverageLoc)> {
        let (start_file, start_line, start_col) =
            lookup_file_line_col(source_map, BytePos::from_u32(self.start_byte_pos));
        let (end_file, end_line, end_col) =
            lookup_file_line_col(source_map, BytePos::from_u32(self.end_byte_pos));
        let start_file_path = match &start_file.name {
            FileName::Real(RealFileName::Named(path)) => path,
            _ => {
                bug!("start_file_path should be a RealFileName, but it was: {:?}", start_file.name)
            }
        };
        let end_file_path = match &end_file.name {
            FileName::Real(RealFileName::Named(path)) => path,
            _ => bug!("end_file_path should be a RealFileName, but it was: {:?}", end_file.name),
        };
        if start_file_path == end_file_path {
            Some((start_file, CoverageLoc { start_line, start_col, end_line, end_col }))
        } else {
            None
            // FIXME(richkadel): There seems to be a problem computing the file location in
            // some cases. I need to investigate this more. When I generate and show coverage
            // for the example binary in the crates.io crate `json5format`, I had a couple of
            // notable problems:
            //
            //   1. I saw a lot of coverage spans in `llvm-cov show` highlighting regions in
            //      various comments (not corresponding to rustdoc code), indicating a possible
            //      problem with the byte_pos-to-source-map implementation.
            //
            //   2. And (perhaps not related) when I build the aforementioned example binary with:
            //      `RUST_FLAGS="-Zinstrument-coverage" cargo build --example formatjson5`
            //      and then run that binary with
            //      `LLVM_PROFILE_FILE="formatjson5.profraw" ./target/debug/examples/formatjson5 \
            //      some.json5` for some reason the binary generates *TWO* `.profraw` files. One
            //      named `default.profraw` and the other named `formatjson5.profraw` (the expected
            //      name, in this case).
            //
            // If the byte range conversion is wrong, fix it. But if it
            // is right, then it is possible for the start and end to be in different files.
            // Can I do something other than ignore coverages that span multiple files?
            //
            // If I can resolve this, remove the "Option<>" result type wrapper
            // `regions_in_file_order()` accordingly.
        }
    }
}

impl Default for CoverageRegion {
    fn default() -> Self {
        Self {
            // The default kind (Unreachable) is a placeholder that will be overwritten before
            // backend codegen.
            kind: CoverageKind::Unreachable,
            start_byte_pos: 0,
            end_byte_pos: 0,
        }
    }
}

/// A source code region used with coverage information.
#[derive(Debug, Eq, PartialEq)]
pub struct CoverageLoc {
    /// The (1-based) line number of the region start.
    pub start_line: u32,
    /// The (1-based) column number of the region start.
    pub start_col: u32,
    /// The (1-based) line number of the region end.
    pub end_line: u32,
    /// The (1-based) column number of the region end.
    pub end_col: u32,
}

impl Ord for CoverageLoc {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.start_line, &self.start_col, &self.end_line, &self.end_col).cmp(&(
            other.start_line,
            &other.start_col,
            &other.end_line,
            &other.end_col,
        ))
    }
}

impl PartialOrd for CoverageLoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for CoverageLoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Customize debug format, and repeat the file name, so generated location strings are
        // "clickable" in many IDEs.
        write!(f, "{}:{} - {}:{}", self.start_line, self.start_col, self.end_line, self.end_col)
    }
}

fn lookup_file_line_col(source_map: &SourceMap, byte_pos: BytePos) -> (Lrc<SourceFile>, u32, u32) {
    let found = source_map
        .lookup_line(byte_pos)
        .expect("should find coverage region byte position in source");
    let file = found.sf;
    let line_pos = file.line_begin_pos(byte_pos);

    // Use 1-based indexing.
    let line = (found.line + 1) as u32;
    let col = (byte_pos - line_pos).to_u32() + 1;

    (file, line, col)
}

/// Collects all of the coverage regions associated with (a) injected counters, (b) counter
/// expressions (additions or subtraction), and (c) unreachable regions (always counted as zero),
/// for a given Function. Counters and counter expressions are indexed because they can be operands
/// in an expression. This struct also stores the `function_source_hash`, computed during
/// instrumentation and forwarded with counters.
///
/// Note, it's important to distinguish the `unreachable` region type from what LLVM's refers to as
/// a "gap region" (or "gap area"). A gap region is a code region within a counted region (either
/// counter or expression), but the line or lines in the gap region are not executable (such as
/// lines with only whitespace or comments). According to LLVM Code Coverage Mapping documentation,
/// "A count for a gap area is only used as the line execution count if there are no other regions
/// on a line."
pub struct FunctionCoverage {
    source_hash: u64,
    counters: Vec<CoverageRegion>,
    expressions: Vec<CoverageRegion>,
    unreachable: Vec<CoverageRegion>,
    translated: bool,
}

impl FunctionCoverage {
    pub fn with_coverageinfo<'tcx>(coverageinfo: &'tcx mir::CoverageInfo) -> Self {
        Self {
            source_hash: 0, // will be set with the first `add_counter()`
            counters: vec![CoverageRegion::default(); coverageinfo.num_counters as usize],
            expressions: vec![CoverageRegion::default(); coverageinfo.num_expressions as usize],
            unreachable: Vec::new(),
            translated: false,
        }
    }

    /// Adds a code region to be counted by an injected counter intrinsic. Return a counter ID
    /// for the call.
    pub fn add_counter(
        &mut self,
        source_hash: u64,
        index: u32,
        start_byte_pos: u32,
        end_byte_pos: u32,
    ) {
        self.source_hash = source_hash;
        self.counters[index as usize] =
            CoverageRegion { kind: CoverageKind::Counter, start_byte_pos, end_byte_pos };
    }

    pub fn add_counter_expression(
        &mut self,
        translated_index: u32,
        lhs: u32,
        op: CounterOp,
        rhs: u32,
        start_byte_pos: u32,
        end_byte_pos: u32,
    ) {
        let index = u32::MAX - translated_index;
        // Counter expressions start with "translated indexes", descending from `u32::MAX`, so
        // the range of expression indexes is disjoint from the range of counter indexes. This way,
        // both counters and expressions can be operands in other expressions.
        //
        // Once all counters have been added, the final "region index" for an expression is
        // `counters.len() + expression_index` (where `expression_index` is its index in
        // `self.expressions`), and the expression operands (`lhs` and `rhs`) can be converted to
        // final "region index" references by the same conversion, after subtracting from
        // `u32::MAX`.
        self.expressions[index as usize] = CoverageRegion {
            kind: CoverageKind::CounterExpression(lhs, op, rhs),
            start_byte_pos,
            end_byte_pos,
        };
    }

    pub fn add_unreachable(&mut self, start_byte_pos: u32, end_byte_pos: u32) {
        self.unreachable.push(CoverageRegion {
            kind: CoverageKind::Unreachable,
            start_byte_pos,
            end_byte_pos,
        });
    }

    pub fn source_hash(&self) -> u64 {
        self.source_hash
    }

    fn regions(&'a mut self) -> impl Iterator<Item = &'a CoverageRegion> {
        assert!(self.source_hash != 0);
        self.ensure_expressions_translated();
        self.counters.iter().chain(self.expressions.iter().chain(self.unreachable.iter()))
    }

    pub fn regions_in_file_order(
        &'a mut self,
        source_map: &SourceMap,
    ) -> BTreeMap<PathBuf, BTreeMap<CoverageLoc, (usize, CoverageKind)>> {
        let mut regions_in_file_order = BTreeMap::new();
        for (region_id, region) in self.regions().enumerate() {
            if let Some((source_file, region_loc)) = region.source_loc(source_map) {
                // FIXME(richkadel): `region.source_loc()` sometimes fails with two different
                // filenames for the start and end byte position. This seems wrong, but for
                // now, if encountered, the region is skipped. If resolved, convert the result
                // to a non-option value so regions are never skipped.
                let real_file_path = match &(*source_file).name {
                    FileName::Real(RealFileName::Named(path)) => path.clone(),
                    _ => bug!("coverage mapping expected only real, named files"),
                };
                let file_coverage_regions =
                    regions_in_file_order.entry(real_file_path).or_insert_with(|| BTreeMap::new());
                file_coverage_regions.insert(region_loc, (region_id, region.kind));
            }
        }
        regions_in_file_order
    }

    /// A one-time translation of expression operands is needed, for any operands referencing
    /// other CounterExpressions. CounterExpression operands get an initial operand ID that is
    /// computed by the simple translation: `u32::max - expression_index` because, when created,
    /// the total number of Counters is not yet known. This function recomputes region indexes
    /// for expressions so they start with the next region index after the last counter index.
    fn ensure_expressions_translated(&mut self) {
        if !self.translated {
            self.translated = true;
            let start = self.counters.len() as u32;
            assert!(
                (start as u64 + self.expressions.len() as u64) < u32::MAX as u64,
                "the number of counters and counter expressions in a single function exceeds {}",
                u32::MAX
            );
            for region in self.expressions.iter_mut() {
                match region.kind {
                    CoverageKind::CounterExpression(lhs, op, rhs) => {
                        let lhs = to_region_index(start, lhs);
                        let rhs = to_region_index(start, rhs);
                        region.kind = CoverageKind::CounterExpression(lhs, op, rhs);
                    }
                    _ => bug!("expressions must only contain CounterExpression kinds"),
                }
            }
        }
    }
}

fn to_region_index(start: u32, index: u32) -> u32 {
    if index < start { index } else { start + (u32::MAX - index) }
}
