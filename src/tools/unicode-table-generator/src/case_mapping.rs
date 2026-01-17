use std::char;
use std::collections::BTreeMap;

use crate::UnicodeData;
use crate::fmt_helpers::{CharEscape, Hex, fmt_list};

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
    let snake_case_name = case.to_lowercase();
    let screaming_case_name = case.to_uppercase();
    let table_name = format!("{screaming_case_name}CASE_TABLE");
    let multi_table_name = format!("{screaming_case_name}CASE_TABLE_MULTI");

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
    let tables = format!(
        "
#[rustfmt::skip]
static {table_name}: &[(char, u32); {mappings_len}] = &[{mappings}];

#[rustfmt::skip]
static {multi_table_name}: &[[char; 3]; {multis_len}] = &[{multis}];

const _: () = {{
    let mut i = 0;
    while i < {table_name}.len() {{
        let (_, val) = {table_name}[i];
        if val & (1 << 22) == 0 {{
            assert!(char::from_u32(val).is_some());
        }} else {{
            let index = val & ((1 << 22) - 1);
            assert!((index as usize) < {multi_table_name}.len());
        }}
        i += 1;
    }}
}};

#[inline]
pub fn to_{snake_case_name}(c: char) -> [char; 3] {{
    // SAFETY: Just checked that the tables are valid
    unsafe {{
        super::case_conversion(
            c,
            |c| c.to_ascii_{snake_case_name}case(),
            {table_name},
            {multi_table_name},
        )
    }}
}}",
        mappings = fmt_list(&mappings),
        mappings_len = mappings.len(),
        multis = fmt_list(&multis),
        multis_len = multis.len(),
    );

    (tables, size)
}
