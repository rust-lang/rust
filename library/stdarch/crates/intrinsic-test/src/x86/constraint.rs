use crate::common::constraint::Constraint;

pub fn map_constraints(imm_type: &String) -> Option<Constraint> {
    match imm_type.as_str() {
        "_MM_FROUND" => Some(Constraint::Range(0..4)),
        "_MM_INDEX_SCALE" => Some(Constraint::Set(vec![1, 2, 4, 8])),
        "_MM_CMPINT" => Some(Constraint::Range(0..8)),
        "_MM_REDUCE" => Some(Constraint::Range(0..8)),
        "_MM_FROUND_SAE" => Some(Constraint::Range(0..8)),
        "_MM_MANTISSA_NORM" => Some(Constraint::Range(0..4)),
        "_MM_MANTISSA_NORM_ENUM" => Some(Constraint::Range(0..4)),
        "_MM_MANTISSA_SIGN" => Some(Constraint::Range(0..3)),
        "_MM_PERM" => Some(Constraint::Range(0..256)),
        "_MM_PERM_ENUM" => Some(Constraint::Range(0..256)),
        "_MM_CMPINT_ENUM" => Some(Constraint::Range(0..8)),
        "_MM_ROUND_MODE" => Some(Constraint::Set(vec![0, 0x2000, 0x4000, 0x6000])),
        "_CMP_" => Some(Constraint::Range(0..32)),
        _ => None,
    }
}