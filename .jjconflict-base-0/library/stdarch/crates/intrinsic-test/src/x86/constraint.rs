use crate::common::constraint::Constraint;

pub fn map_constraints(fn_name: &str, imm_type: &String, imm_width: u32) -> Option<Constraint> {
    if imm_width > 0 {
        if fn_name == "_mm_sm3rnds2_epi32" {
            return Some(Constraint::Set((0..64).step_by(2).collect()));
        }
        let max: i64 = 2i64.pow(imm_width);
        return Some(Constraint::Range(0..max));
    }
    match imm_type.as_str() {
        // _mm512_cvt{_round}ps_ph functions can accept a larger set of values for _MM_FROUND
        "_MM_FROUND"
            if fn_name.starts_with("_mm512")
                && (fn_name.ends_with("cvtps_ph") || fn_name.ends_with("cvt_roundps_ph")) =>
        {
            Some(Constraint::Set(vec![0, 1, 2, 3, 4, 8, 9, 10, 11, 12]))
        }
        "_MM_FROUND" => Some(Constraint::Set(vec![4, 8, 9, 10, 11])),
        "_MM_INDEX_SCALE" => Some(Constraint::Set(vec![1, 2, 4, 8])),
        "_MM_CMPINT" => Some(Constraint::Range(0..8)),
        "_MM_REDUCE" => Some(Constraint::Range(0..256)),
        "_MM_FROUND_SAE" => Some(Constraint::Set(vec![4, 8])),
        "_MM_MANTISSA_NORM" => Some(Constraint::Range(0..4)),
        "_MM_MANTISSA_SIGN" => Some(Constraint::Range(0..3)),
        "_MM_PERM" => Some(Constraint::Range(0..256)),
        "_MM_ROUND_MODE" => Some(Constraint::Range(0..5)),
        "_CMP_" => Some(Constraint::Range(0..32)),
        _ => None,
    }
}
