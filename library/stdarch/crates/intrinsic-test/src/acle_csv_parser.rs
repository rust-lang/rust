use itertools::Itertools;
use regex::Regex;
use serde::Deserialize;

use crate::argument::{Argument, ArgumentList, Constraint};
use crate::intrinsic::Intrinsic;
use crate::types::{IntrinsicType, TypeKind};

pub fn get_acle_intrinsics(filename: &str) -> Vec<Intrinsic> {
    let data = std::fs::read_to_string(filename).expect("Failed to open ACLE intrinsics file");

    let data = data
        .lines()
        .filter_map(|l| {
            (!(l.starts_with("<COMMENT>") || l.is_empty() || l.starts_with("<SECTION>")))
                .then(|| l.replace("<HEADER>\t", ""))
        })
        .join("\n");

    let mut csv_reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(data.as_bytes());

    let mut intrinsics: Vec<Intrinsic> = csv_reader
        .deserialize()
        .filter_map(|x: Result<ACLEIntrinsicLine, _>| x.ok().map(|i| i.into()))
        .collect::<Vec<_>>();

    // Intrinsics such as vshll_n_s8 exist twice in the ACLE with different constraints.
    intrinsics.sort_by(|a, b| a.name.cmp(&b.name));
    let (intrinsics, duplicates) = intrinsics.partition_dedup_by(|a, b| a.name == b.name);
    for duplicate in duplicates {
        let name = &duplicate.name;
        let constraints = duplicate
            .arguments
            .args
            .drain(..)
            .filter(|a| a.has_constraint());
        let intrinsic = intrinsics.iter_mut().find(|i| &i.name == name).unwrap();

        for mut constraint in constraints {
            let real_constraint = intrinsic
                .arguments
                .args
                .iter_mut()
                .find(|a| a.name == constraint.name)
                .unwrap();
            real_constraint
                .constraints
                .push(constraint.constraints.pop().unwrap());
        }
    }

    intrinsics.to_vec()
}

impl Into<Intrinsic> for ACLEIntrinsicLine {
    fn into(self) -> Intrinsic {
        let signature = self.intrinsic;
        let (ret_ty, remaining) = signature.split_once(' ').unwrap();

        let results = type_from_c(ret_ty)
            .unwrap_or_else(|_| panic!("Failed to parse return type: {}", ret_ty));

        let (name, args) = remaining.split_once('(').unwrap();
        let args = args.trim_end_matches(')');

        // Typo in ACLE data
        let args = args.replace("int16x8q_t", "int16x8_t");

        let arg_prep = self.argument_preparation.as_str();
        let args = args
            .split(',')
            .enumerate()
            .map(move |(idx, arg)| {
                let arg = arg.trim();
                if arg.starts_with("__builtin_constant_p") {
                    handle_constraint(idx, arg, arg_prep)
                } else {
                    from_c(idx, arg)
                }
            })
            .collect();
        let arguments = ArgumentList { args };

        Intrinsic {
            name: name.to_string(),
            arguments,
            results,
        }
    }
}

fn handle_constraint(idx: usize, arg: &str, prep: &str) -> Argument {
    let prep = prep.replace(' ', "");

    let name = arg
        .trim_start_matches("__builtin_constant_p")
        .trim_start_matches(|ref c| c == &' ' || c == &'(')
        .trim_end_matches(')')
        .to_string();

    let ty = IntrinsicType::Type {
        constant: true,
        kind: TypeKind::Int,
        bit_len: Some(32),
        simd_len: None,
        vec_len: None,
    };

    let constraints = prep
        .split(';')
        .find_map(|p| handle_range_constraint(&name, p).or_else(|| handle_eq_constraint(&name, p)))
        .map(|c| vec![c])
        .unwrap_or_default();

    Argument {
        pos: idx,
        name,
        ty,
        constraints,
    }
}

fn handle_range_constraint(name: &str, data: &str) -> Option<Constraint> {
    lazy_static! {
        static ref RANGE_CONSTRAINT: Regex =
            Regex::new(r#"([0-9]+)<=([[:alnum:]]+)<=([0-9]+)"#).unwrap();
    }

    let captures = RANGE_CONSTRAINT.captures(data)?;
    if captures.get(2).map(|c| c.as_str() == name).unwrap_or(false) {
        match (captures.get(1), captures.get(3)) {
            (Some(start), Some(end)) => {
                let start = start.as_str().parse::<i64>().unwrap();
                let end = end.as_str().parse::<i64>().unwrap() + 1;
                Some(Constraint::Range(start..end))
            }
            _ => panic!("Invalid constraint"),
        }
    } else {
        None
    }
}

fn handle_eq_constraint(name: &str, data: &str) -> Option<Constraint> {
    lazy_static! {
        static ref EQ_CONSTRAINT: Regex = Regex::new(r#"([[:alnum:]]+)==([0-9]+)"#).unwrap();
    }
    let captures = EQ_CONSTRAINT.captures(data)?;
    if captures.get(1).map(|c| c.as_str() == name).unwrap_or(false) {
        captures
            .get(2)
            .map(|c| Constraint::Equal(c.as_str().parse::<i64>().unwrap()))
    } else {
        None
    }
}

fn from_c(pos: usize, s: &str) -> Argument {
    let name_index = s
        .chars()
        .rev()
        .take_while(|c| c != &'*' && c != &' ')
        .count();

    let name_start = s.len() - name_index;
    let name = s[name_start..].to_string();
    let s = s[..name_start].trim();

    Argument {
        pos,
        name,
        ty: type_from_c(s).unwrap_or_else(|_| panic!("Failed to parse type: {}", s)),
        constraints: vec![],
    }
}

fn type_from_c(s: &str) -> Result<IntrinsicType, String> {
    const CONST_STR: &str = "const ";

    if let Some(s) = s.strip_suffix('*') {
        let (s, constant) = if s.ends_with(CONST_STR) {
            (&s[..s.len() - (CONST_STR.len() + 1)], true)
        } else {
            (s, false)
        };

        let s = s.trim_end();

        Ok(IntrinsicType::Ptr {
            constant,
            child: Box::new(type_from_c(s)?),
        })
    } else {
        // [const ]TYPE[{bitlen}[x{simdlen}[x{vec_len}]]][_t]

        let (mut s, constant) = if let Some(s) = s.strip_prefix(CONST_STR) {
            (s, true)
        } else {
            (s, false)
        };
        s = s.strip_suffix("_t").unwrap_or(s);

        let mut parts = s.split('x'); // [[{bitlen}], [{simdlen}], [{vec_len}] ]

        let start = parts.next().ok_or("Impossible to parse type")?;

        if let Some(digit_start) = start.find(|c: char| c.is_ascii_digit()) {
            let (arg_kind, bit_len) = start.split_at(digit_start);

            let arg_kind = arg_kind.parse::<TypeKind>()?;
            let bit_len = bit_len.parse::<u32>().map_err(|err| err.to_string())?;

            let simd_len = parts.next().map(|part| part.parse::<u32>().ok()).flatten();
            let vec_len = parts.next().map(|part| part.parse::<u32>().ok()).flatten();

            Ok(IntrinsicType::Type {
                constant,
                kind: arg_kind,
                bit_len: Some(bit_len),
                simd_len,
                vec_len,
            })
        } else {
            Ok(IntrinsicType::Type {
                constant,
                kind: start.parse::<TypeKind>()?,
                bit_len: None,
                simd_len: None,
                vec_len: None,
            })
        }
    }
}

#[derive(Deserialize, Debug, PartialEq, Clone)]
struct ACLEIntrinsicLine {
    #[serde(rename = "Intrinsic")]
    intrinsic: String,
    #[serde(rename = "Argument preparation")]
    argument_preparation: String,
    #[serde(rename = "AArch64 Instruction")]
    aarch64_instruction: String,
    #[serde(rename = "Result")]
    result: String,
    #[serde(rename = "Supported architectures")]
    supported_architectures: String,
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::argument::Argument;
    use crate::types::{IntrinsicType, TypeKind};

    #[test]
    fn parse_simd() {
        let expected = Argument {
            pos: 0,
            name: "a".into(),
            ty: IntrinsicType::Type {
                constant: false,
                kind: TypeKind::Int,
                bit_len: Some(32),
                simd_len: Some(4),
                vec_len: None,
            },
            constraints: vec![],
        };
        let actual = from_c(0, "int32x4_t a");
        assert_eq!(expected, actual);
    }

    #[test]
    fn parse_simd_with_vec() {
        let expected = Argument {
            pos: 0,
            name: "a".into(),
            ty: IntrinsicType::Type {
                constant: false,
                kind: TypeKind::Int,
                bit_len: Some(32),
                simd_len: Some(4),
                vec_len: Some(2),
            },
            constraints: vec![],
        };
        let actual = from_c(0, "int32x4x2_t a");
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_ptr() {
        let expected = Argument {
            pos: 0,
            name: "ptr".into(),
            ty: crate::types::IntrinsicType::Ptr {
                constant: true,
                child: Box::new(IntrinsicType::Type {
                    constant: false,
                    kind: TypeKind::Int,
                    bit_len: Some(8),
                    simd_len: None,
                    vec_len: None,
                }),
            },
            constraints: vec![],
        };
        let actual = from_c(0, "int8_t const *ptr");
        assert_eq!(expected, actual);
    }
}
