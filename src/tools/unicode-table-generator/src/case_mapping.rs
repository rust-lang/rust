use std::char;
use std::collections::BTreeMap;
use std::fmt::{self, Write};

use crate::{UnicodeData, fmt_list};

const INDEX_MASK: u32 = 1 << 22;

pub(crate) fn generate_case_mapping(data: &UnicodeData) -> String {
    let mut file = String::new();

    write!(file, "const INDEX_MASK: u32 = 0x{:x};", INDEX_MASK).unwrap();
    file.push_str("\n\n");
    file.push_str(HEADER.trim_start());
    file.push('\n');
    file.push_str(&generate_tables("LOWER", &data.to_lower));
    file.push_str("\n\n");
    file.push_str(&generate_tables("UPPER", &data.to_upper));
    file
}

fn generate_tables(case: &str, data: &BTreeMap<u32, (u32, u32, u32)>) -> String {
    let mut mappings = Vec::with_capacity(data.len());
    let mut multis = Vec::new();

    for (&key, &(a, b, c)) in data.iter() {
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

        mappings.push((CharEscape(key), value));
    }

    let mut tables = String::new();

    write!(tables, "static {}CASE_TABLE: &[(char, u32)] = &[{}];", case, fmt_list(mappings))
        .unwrap();

    tables.push_str("\n\n");

    write!(tables, "static {}CASE_TABLE_MULTI: &[[char; 3]] = &[{}];", case, fmt_list(multis))
        .unwrap();

    tables
}

struct CharEscape(char);

impl fmt::Debug for CharEscape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "'{}'", self.0.escape_default())
    }
}

static HEADER: &str = r"
pub fn to_lower(c: char) -> [char; 3] {
    if c.is_ascii() {
        [(c as u8).to_ascii_lowercase() as char, '\0', '\0']
    } else {
        LOWERCASE_TABLE
            .binary_search_by(|&(key, _)| key.cmp(&c))
            .map(|i| {
                let u = LOWERCASE_TABLE[i].1;
                char::from_u32(u).map(|c| [c, '\0', '\0']).unwrap_or_else(|| {
                    // SAFETY: Index comes from statically generated table
                    unsafe { *LOWERCASE_TABLE_MULTI.get_unchecked((u & (INDEX_MASK - 1)) as usize) }
                })
            })
            .unwrap_or([c, '\0', '\0'])
    }
}

pub fn to_upper(c: char) -> [char; 3] {
    if c.is_ascii() {
        [(c as u8).to_ascii_uppercase() as char, '\0', '\0']
    } else {
        UPPERCASE_TABLE
            .binary_search_by(|&(key, _)| key.cmp(&c))
            .map(|i| {
                let u = UPPERCASE_TABLE[i].1;
                char::from_u32(u).map(|c| [c, '\0', '\0']).unwrap_or_else(|| {
                    // SAFETY: Index comes from statically generated table
                    unsafe { *UPPERCASE_TABLE_MULTI.get_unchecked((u & (INDEX_MASK - 1)) as usize) }
                })
            })
            .unwrap_or([c, '\0', '\0'])
    }
}
";
