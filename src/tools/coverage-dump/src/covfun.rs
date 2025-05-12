use std::collections::HashMap;
use std::fmt::{self, Debug, Write as _};
use std::sync::LazyLock;

use anyhow::{Context, anyhow, bail, ensure};
use itertools::Itertools;
use regex::Regex;

use crate::covmap::FilenameTables;
use crate::llvm_utils::unescape_llvm_string_contents;
use crate::parser::Parser;

#[cfg(test)]
mod tests;

pub(crate) fn dump_covfun_mappings(
    llvm_ir: &str,
    filename_tables: &FilenameTables,
    function_names: &HashMap<u64, String>,
) -> anyhow::Result<()> {
    // Extract function coverage entries from the LLVM IR assembly, and associate
    // each entry with its (demangled) name.
    let mut covfun_entries = llvm_ir
        .lines()
        .filter(|line| is_covfun_line(line))
        .map(parse_covfun_line)
        .map_ok(|line_data| {
            (function_names.get(&line_data.name_hash).map(String::as_str), line_data)
        })
        .collect::<Result<Vec<_>, _>>()?;
    covfun_entries.sort_by(|a, b| {
        // Sort entries primarily by name, to help make the order consistent
        // across platforms and relatively insensitive to changes.
        // (Sadly we can't use `sort_by_key` because we would need to return references.)
        Ord::cmp(&a.0, &b.0)
            .then_with(|| Ord::cmp(&a.1.is_used, &b.1.is_used))
            .then_with(|| Ord::cmp(a.1.payload.as_slice(), b.1.payload.as_slice()))
    });

    for (name, line_data) in &covfun_entries {
        let name = name.unwrap_or("(unknown)");
        let unused = if line_data.is_used { "" } else { " (unused)" };
        println!("Function name: {name}{unused}");

        let payload: &[u8] = &line_data.payload;
        println!("Raw bytes ({len}): 0x{payload:02x?}", len = payload.len());

        let mut parser = Parser::new(payload);

        let num_files = parser.read_uleb128_u32()?;
        println!("Number of files: {num_files}");

        for i in 0..num_files {
            let global_file_id = parser.read_uleb128_usize()?;
            let &CovfunLineData { filenames_hash, .. } = line_data;
            let Some(filename) = filename_tables.lookup(filenames_hash, global_file_id) else {
                bail!("couldn't resolve global file: {filenames_hash}, {global_file_id}");
            };
            println!("- file {i} => {filename}");
        }

        let num_expressions = parser.read_uleb128_u32()?;
        println!("Number of expressions: {num_expressions}");

        let mut expression_resolver = ExpressionResolver::new();
        for i in 0..num_expressions {
            let lhs = parser.read_simple_term()?;
            let rhs = parser.read_simple_term()?;
            println!("- expression {i} operands: lhs = {lhs:?}, rhs = {rhs:?}");
            expression_resolver.push_operands(lhs, rhs);
        }

        let mut max_counter = None;
        for i in 0..num_files {
            let num_mappings = parser.read_uleb128_u32()?;
            println!("Number of file {i} mappings: {num_mappings}");

            for _ in 0..num_mappings {
                let (kind, region) = parser.read_mapping_kind_and_region()?;
                println!("- {kind:?} at {region:?}");
                kind.for_each_term(|term| {
                    if let CovTerm::Counter(n) = term {
                        max_counter = max_counter.max(Some(n));
                    }
                });

                match kind {
                    // Also print expression mappings in resolved form.
                    MappingKind::Code(term @ CovTerm::Expression { .. })
                    | MappingKind::Gap(term @ CovTerm::Expression { .. }) => {
                        println!("    = {}", expression_resolver.format_term(term));
                    }
                    // If the mapping is a branch region, print both of its arms
                    // in resolved form (even if they aren't expressions).
                    MappingKind::Branch { r#true, r#false }
                    | MappingKind::MCDCBranch { r#true, r#false, .. } => {
                        println!("    true  = {}", expression_resolver.format_term(r#true));
                        println!("    false = {}", expression_resolver.format_term(r#false));
                    }
                    _ => (),
                }
            }
        }

        parser.ensure_empty()?;

        // Printing the highest counter ID seen in the functions mappings makes
        // it easier to determine whether a change to coverage instrumentation
        // has increased or decreased the number of physical counters needed.
        // (It's possible for the generated code to have more counters that
        // aren't used by any mappings, but that should hopefully be rare.)
        println!(
            "Highest counter ID seen: {}",
            match max_counter {
                Some(id) => format!("c{id}"),
                None => "(none)".to_owned(),
            }
        );
        println!();
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct CovfunLineData {
    is_used: bool,
    name_hash: u64,
    filenames_hash: u64,
    payload: Vec<u8>,
}

fn is_covfun_line(line: &str) -> bool {
    line.starts_with("@__covrec_")
}

/// Given a line of LLVM IR assembly that should contain an `__llvm_covfun`
/// entry, parses it to extract relevant data in a `CovfunLineData`.
fn parse_covfun_line(line: &str) -> anyhow::Result<CovfunLineData> {
    ensure!(is_covfun_line(line));

    // We cheat a little bit and match variable names `@__covrec_[HASH]u`
    // rather than the section name, because the section name is harder to
    // extract and differs across Linux/Windows/macOS.
    const RE_STRING: &str = r#"(?x)^
        @__covrec_[0-9A-Z]+(?<is_used>u)?
        \ = \ # (trailing space)
        .*
        <\{
            \ i64 \ (?<name_hash> -? [0-9]+),
            \ i32 \ -? [0-9]+, # (length of payload; currently unused)
            \ i64 \ -? [0-9]+, # (source hash; currently unused)
            \ i64 \ (?<filenames_hash> -? [0-9]+),
            \ \[ [0-9]+ \ x \ i8 \] \ c"(?<payload>[^"]*)"
            \ # (trailing space)
        }>
        .*$
    "#;
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(RE_STRING).unwrap());

    let captures =
        RE.captures(line).with_context(|| format!("couldn't parse covfun line: {line:?}"))?;
    let is_used = captures.name("is_used").is_some();
    let name_hash = i64::from_str_radix(&captures["name_hash"], 10).unwrap() as u64;
    let filenames_hash = i64::from_str_radix(&captures["filenames_hash"], 10).unwrap() as u64;
    let payload = unescape_llvm_string_contents(&captures["payload"]);

    Ok(CovfunLineData { is_used, name_hash, filenames_hash, payload })
}

// Extra parser methods only needed when parsing `covfun` payloads.
impl<'a> Parser<'a> {
    fn read_simple_term(&mut self) -> anyhow::Result<CovTerm> {
        let raw_term = self.read_uleb128_u32()?;
        CovTerm::decode(raw_term).context("decoding term")
    }

    fn read_mapping_kind_and_region(&mut self) -> anyhow::Result<(MappingKind, MappingRegion)> {
        let mut kind = self.read_raw_mapping_kind()?;
        let mut region = self.read_raw_mapping_region()?;

        const HIGH_BIT: u32 = 1u32 << 31;
        if region.end_column & HIGH_BIT != 0 {
            region.end_column &= !HIGH_BIT;
            kind = match kind {
                MappingKind::Code(term) => MappingKind::Gap(term),
                // LLVM's coverage mapping reader will actually handle this
                // case without complaint, but the result is almost certainly
                // a meaningless implementation artifact.
                _ => return Err(anyhow!("unexpected base kind for gap region: {kind:?}")),
            }
        }

        Ok((kind, region))
    }

    fn read_raw_mapping_kind(&mut self) -> anyhow::Result<MappingKind> {
        let raw_mapping_kind = self.read_uleb128_u32()?;
        if let Some(term) = CovTerm::decode(raw_mapping_kind) {
            return Ok(MappingKind::Code(term));
        }

        assert_eq!(raw_mapping_kind & 0b11, 0);
        assert_ne!(raw_mapping_kind, 0);

        let (high, is_expansion) = (raw_mapping_kind >> 3, raw_mapping_kind & 0b100 != 0);
        if is_expansion {
            Ok(MappingKind::Expansion(high))
        } else {
            match high {
                0 => unreachable!("zero kind should have already been handled as a code mapping"),
                2 => Ok(MappingKind::Skip),
                4 => {
                    let r#true = self.read_simple_term()?;
                    let r#false = self.read_simple_term()?;
                    Ok(MappingKind::Branch { r#true, r#false })
                }
                5 => {
                    let bitmap_idx = self.read_uleb128_u32()?;
                    let conditions_num = self.read_uleb128_u32()?;
                    Ok(MappingKind::MCDCDecision { bitmap_idx, conditions_num })
                }
                6 => {
                    let r#true = self.read_simple_term()?;
                    let r#false = self.read_simple_term()?;
                    let condition_id = self.read_uleb128_u32()?;
                    let true_next_id = self.read_uleb128_u32()?;
                    let false_next_id = self.read_uleb128_u32()?;
                    Ok(MappingKind::MCDCBranch {
                        r#true,
                        r#false,
                        condition_id,
                        true_next_id,
                        false_next_id,
                    })
                }

                _ => Err(anyhow!("unknown mapping kind: {raw_mapping_kind:#x}")),
            }
        }
    }

    fn read_raw_mapping_region(&mut self) -> anyhow::Result<MappingRegion> {
        let start_line_offset = self.read_uleb128_u32()?;
        let start_column = self.read_uleb128_u32()?;
        let end_line_offset = self.read_uleb128_u32()?;
        let end_column = self.read_uleb128_u32()?;
        Ok(MappingRegion { start_line_offset, start_column, end_line_offset, end_column })
    }
}

/// Enum that can hold a constant zero value, the ID of an physical coverage
/// counter, or the ID (and operation) of a coverage-counter expression.
///
/// Terms are used as the operands of coverage-counter expressions, as the arms
/// of branch mappings, and as the value of code/gap mappings.
#[derive(Clone, Copy, Debug)]
pub(crate) enum CovTerm {
    Zero,
    Counter(u32),
    Expression(u32, Op),
}

/// Operator (addition or subtraction) used by an expression.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Op {
    Sub,
    Add,
}

impl CovTerm {
    pub(crate) fn decode(input: u32) -> Option<Self> {
        let (high, tag) = (input >> 2, input & 0b11);
        match tag {
            0b00 if high == 0 => Some(Self::Zero),
            0b01 => Some(Self::Counter(high)),
            0b10 => Some(Self::Expression(high, Op::Sub)),
            0b11 => Some(Self::Expression(high, Op::Add)),
            // When reading expression operands or branch arms, the LLVM coverage
            // mapping reader will always interpret a `0b00` tag as a zero
            // term, even when the high bits are non-zero.
            // We treat that case as failure instead, so that this code can be
            // shared by the full mapping-kind reader as well.
            _ => None,
        }
    }
}

#[derive(Debug)]
enum MappingKind {
    Code(CovTerm),
    Gap(CovTerm),
    Expansion(#[allow(dead_code)] u32),
    Skip,
    // Using raw identifiers here makes the dump output a little bit nicer
    // (via the derived Debug), at the expense of making this tool's source
    // code a little bit uglier.
    Branch {
        r#true: CovTerm,
        r#false: CovTerm,
    },
    MCDCBranch {
        r#true: CovTerm,
        r#false: CovTerm,
        // These attributes are printed in Debug but not used directly.
        #[allow(dead_code)]
        condition_id: u32,
        #[allow(dead_code)]
        true_next_id: u32,
        #[allow(dead_code)]
        false_next_id: u32,
    },
    MCDCDecision {
        // These attributes are printed in Debug but not used directly.
        #[allow(dead_code)]
        bitmap_idx: u32,
        #[allow(dead_code)]
        conditions_num: u32,
    },
}

impl MappingKind {
    fn for_each_term(&self, mut callback: impl FnMut(CovTerm)) {
        match *self {
            Self::Code(term) => callback(term),
            Self::Gap(term) => callback(term),
            Self::Expansion(_id) => {}
            Self::Skip => {}
            Self::Branch { r#true, r#false } => {
                callback(r#true);
                callback(r#false);
            }
            Self::MCDCBranch {
                r#true,
                r#false,
                condition_id: _,
                true_next_id: _,
                false_next_id: _,
            } => {
                callback(r#true);
                callback(r#false);
            }
            Self::MCDCDecision { bitmap_idx: _, conditions_num: _ } => {}
        }
    }
}

struct MappingRegion {
    /// Offset of this region's start line, relative to the *start line* of
    /// the *previous mapping* (or 0). Line numbers are 1-based.
    start_line_offset: u32,
    /// This region's start column, absolute and 1-based.
    start_column: u32,
    /// Offset of this region's end line, relative to the *this mapping's*
    /// start line. Line numbers are 1-based.
    end_line_offset: u32,
    /// This region's end column, absolute, 1-based, and exclusive.
    ///
    /// If the highest bit is set, that bit is cleared and the associated
    /// mapping becomes a gap region mapping.
    end_column: u32,
}

impl Debug for MappingRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(prev + {}, {}) to (start + {}, {})",
            self.start_line_offset, self.start_column, self.end_line_offset, self.end_column
        )
    }
}

/// Helper type that prints expressions in a "resolved" form, so that
/// developers reading the dump don't need to resolve expressions by hand.
struct ExpressionResolver {
    operands: Vec<(CovTerm, CovTerm)>,
}

impl ExpressionResolver {
    fn new() -> Self {
        Self { operands: Vec::new() }
    }

    fn push_operands(&mut self, lhs: CovTerm, rhs: CovTerm) {
        self.operands.push((lhs, rhs));
    }

    fn format_term(&self, term: CovTerm) -> String {
        let mut output = String::new();
        self.write_term(&mut output, term);
        output
    }

    fn write_term(&self, output: &mut String, term: CovTerm) {
        match term {
            CovTerm::Zero => output.push_str("Zero"),
            CovTerm::Counter(id) => write!(output, "c{id}").unwrap(),
            CovTerm::Expression(id, op) => {
                let (lhs, rhs) = self.operands[id as usize];
                let op = match op {
                    Op::Sub => "-",
                    Op::Add => "+",
                };

                output.push('(');
                self.write_term(output, lhs);
                write!(output, " {op} ").unwrap();
                self.write_term(output, rhs);
                output.push(')');
            }
        }
    }
}
