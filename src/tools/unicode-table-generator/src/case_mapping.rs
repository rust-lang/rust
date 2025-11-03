use std::char;
use std::collections::BTreeMap;

use crate::fmt_helpers::Hex;
use crate::{CharEscape, UnicodeData, fmt_list};

const INDEX_MASK: u32 = 1 << 22;

pub(crate) fn generate_case_mapping(data: &UnicodeData) -> (String, [usize; 2]) {
    let (lower_tables, lower_size) = generate_tables("LOWER", &data.to_lower);
    let (upper_tables, upper_size) = generate_tables("UPPER", &data.to_upper);
    let file = format!(
        "{lower_tables}
        {upper_tables}"
    );
    (file, [lower_size, upper_size])
}

fn generate_tables(case: &str, data: &BTreeMap<u32, [u32; 3]>) -> (String, usize) {
    let case_lower = case.to_lowercase();
    let case_upper = case.to_uppercase();

    let mut mappings = Vec::with_capacity(data.len());
    let mut multis = Vec::new();

    for (&key, &[a, b, c]) in data.iter() {
        let key = char::from_u32(key).unwrap();

        if key.is_ascii() {
            continue;
        }

        let value = if b == 0 && c == 0 {
            a
        } else {
            multis.push([
                CharEscape(char::from_u32(a).unwrap()),
                CharEscape(char::from_u32(b).unwrap()),
                CharEscape(char::from_u32(c).unwrap()),
            ]);

            INDEX_MASK | (u32::try_from(multis.len()).unwrap() - 1)
        };

        mappings.push((CharEscape(key), Hex(value)));
    }

    let size = size_of_val(mappings.as_slice()) + size_of_val(multis.as_slice());
    let file = format!(
        "
#[rustfmt::skip]
static {case}CASE_TABLE: &[(char, u32); {mappings_len}] = &[{mappings}];

#[rustfmt::skip]
static {case}CASE_TABLE_MULTI: &[[char; 3]; {multis_len}] = &[{multis}];

#[inline]
pub fn to_{case_lower}(c: char) -> [char; 3] {{
    const {{
        let mut i = 0;
        while i < {case_upper}CASE_TABLE.len() {{
            let (_, val) = {case_upper}CASE_TABLE[i];
            if val & (1 << 22) == 0 {{
                assert!(char::from_u32(val).is_some());
            }} else {{
                let index = val & ((1 << 22) - 1);
                assert!((index as usize) < {case_upper}CASE_TABLE_MULTI.len());
            }}
            i += 1;
        }}
    }}

    // SAFETY: Just checked that the tables are valid
    unsafe {{
        super::case_conversion(
            c,
            |c| c.to_ascii_{case_lower}case(),
            {case_upper}CASE_TABLE,
            {case_upper}CASE_TABLE_MULTI,
        )
    }}
}}",
        mappings = fmt_list(&mappings),
        mappings_len = mappings.len(),
        multis = fmt_list(&multis),
        multis_len = multis.len(),
    );

    (file, size)
}
