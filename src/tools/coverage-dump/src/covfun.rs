use crate::parser::{unescape_llvm_string_contents, Parser};
use anyhow::{anyhow, Context};
use regex::Regex;
use std::collections::HashMap;
use std::fmt::{self, Debug, Write as _};
use std::sync::OnceLock;

pub(crate) fn dump_covfun_mappings(
    llvm_ir: &str,
    function_names: &HashMap<u64, String>,
) -> anyhow::Result<()> {
    // Extract function coverage entries from the LLVM IR assembly, and associate
    // each entry with its (demangled) name.
    let mut covfun_entries = llvm_ir
        .lines()
        .filter_map(covfun_line_data)
        .map(|line_data| (function_names.get(&line_data.name_hash).map(String::as_str), line_data))
        .collect::<Vec<_>>();
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
            let global_file_id = parser.read_uleb128_u32()?;
            println!("- file {i} => global file {global_file_id}");
        }

        let num_expressions = parser.read_uleb128_u32()?;
        println!("Number of expressions: {num_expressions}");

        let mut expression_resolver = ExpressionResolver::new();
        for i in 0..num_expressions {
            let lhs = parser.read_simple_operand()?;
            let rhs = parser.read_simple_operand()?;
            println!("- expression {i} operands: lhs = {lhs:?}, rhs = {rhs:?}");
            expression_resolver.push_operands(lhs, rhs);
        }

        for i in 0..num_files {
            let num_mappings = parser.read_uleb128_u32()?;
            println!("Number of file {i} mappings: {num_mappings}");

            for _ in 0..num_mappings {
                let (kind, region) = parser.read_mapping_kind_and_region()?;
                println!("- {kind:?} at {region:?}");

                // If the mapping contains expressions, also print the resolved
                // form of those expressions
                kind.for_each_operand(|label, operand| {
                    if matches!(operand, Operand::Expression { .. }) {
                        let pad = if label.is_empty() { "" } else { " " };
                        let resolved = expression_resolver.format_operand(operand);
                        println!("    {label}{pad}= {resolved}");
                    }
                });
            }
        }

        parser.ensure_empty()?;
        println!();
    }
    Ok(())
}

struct CovfunLineData {
    name_hash: u64,
    is_used: bool,
    payload: Vec<u8>,
}

/// Checks a line of LLVM IR assembly to see if it contains an `__llvm_covfun`
/// entry, and if so extracts relevant data in a `CovfunLineData`.
fn covfun_line_data(line: &str) -> Option<CovfunLineData> {
    let re = {
        // We cheat a little bit and match variable names `@__covrec_[HASH]u`
        // rather than the section name, because the section name is harder to
        // extract and differs across Linux/Windows/macOS. We also extract the
        // symbol name hash from the variable name rather than the data, since
        // it's easier and both should match.
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(
                r#"^@__covrec_(?<name_hash>[0-9A-Z]+)(?<is_used>u)? = .*\[[0-9]+ x i8\] c"(?<payload>[^"]*)".*$"#,
            )
            .unwrap()
        })
    };

    let captures = re.captures(line)?;
    let name_hash = u64::from_str_radix(&captures["name_hash"], 16).unwrap();
    let is_used = captures.name("is_used").is_some();
    let payload = unescape_llvm_string_contents(&captures["payload"]);

    Some(CovfunLineData { name_hash, is_used, payload })
}

// Extra parser methods only needed when parsing `covfun` payloads.
impl<'a> Parser<'a> {
    fn read_simple_operand(&mut self) -> anyhow::Result<Operand> {
        let raw_operand = self.read_uleb128_u32()?;
        Operand::decode(raw_operand).context("decoding operand")
    }

    fn read_mapping_kind_and_region(&mut self) -> anyhow::Result<(MappingKind, MappingRegion)> {
        let mut kind = self.read_raw_mapping_kind()?;
        let mut region = self.read_raw_mapping_region()?;

        const HIGH_BIT: u32 = 1u32 << 31;
        if region.end_column & HIGH_BIT != 0 {
            region.end_column &= !HIGH_BIT;
            kind = match kind {
                MappingKind::Code(operand) => MappingKind::Gap(operand),
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
        if let Some(operand) = Operand::decode(raw_mapping_kind) {
            return Ok(MappingKind::Code(operand));
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
                    let r#true = self.read_simple_operand()?;
                    let r#false = self.read_simple_operand()?;
                    Ok(MappingKind::Branch { r#true, r#false })
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

// Represents an expression operand (lhs/rhs), branch region operand (true/false),
// or the value used by a code region or gap region.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Operand {
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

impl Operand {
    pub(crate) fn decode(input: u32) -> Option<Self> {
        let (high, tag) = (input >> 2, input & 0b11);
        match tag {
            0b00 if high == 0 => Some(Self::Zero),
            0b01 => Some(Self::Counter(high)),
            0b10 => Some(Self::Expression(high, Op::Sub)),
            0b11 => Some(Self::Expression(high, Op::Add)),
            // When reading expression or branch operands, the LLVM coverage
            // mapping reader will always interpret a `0b00` tag as a zero
            // operand, even when the high bits are non-zero.
            // We treat that case as failure instead, so that this code can be
            // shared by the full mapping-kind reader as well.
            _ => None,
        }
    }
}

#[derive(Debug)]
enum MappingKind {
    Code(Operand),
    Gap(Operand),
    Expansion(u32),
    Skip,
    Branch { r#true: Operand, r#false: Operand },
}

impl MappingKind {
    /// Visits each operand directly contained in this mapping, along with
    /// a string label (possibly empty).
    fn for_each_operand(&self, mut func: impl FnMut(&str, Operand)) {
        match *self {
            Self::Code(operand) => func("", operand),
            Self::Gap(operand) => func("", operand),
            Self::Expansion(_) => (),
            Self::Skip => (),
            Self::Branch { r#true, r#false } => {
                func("true ", r#true);
                func("false", r#false);
            }
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
    operands: Vec<(Operand, Operand)>,
}

impl ExpressionResolver {
    fn new() -> Self {
        Self { operands: Vec::new() }
    }

    fn push_operands(&mut self, lhs: Operand, rhs: Operand) {
        self.operands.push((lhs, rhs));
    }

    fn format_operand(&self, operand: Operand) -> String {
        let mut output = String::new();
        self.write_operand(&mut output, operand);
        output
    }

    fn write_operand(&self, output: &mut String, operand: Operand) {
        match operand {
            Operand::Zero => output.push_str("Zero"),
            Operand::Counter(id) => write!(output, "c{id}").unwrap(),
            Operand::Expression(id, op) => {
                let (lhs, rhs) = self.operands[id as usize];
                let op = match op {
                    Op::Sub => "-",
                    Op::Add => "+",
                };

                output.push('(');
                self.write_operand(output, lhs);
                write!(output, " {op} ").unwrap();
                self.write_operand(output, rhs);
                output.push(')');
            }
        }
    }
}
