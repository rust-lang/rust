use crate::{fmt_list, UnicodeData};
use std::{
    collections::BTreeMap,
    fmt::{self, Write},
};

pub(crate) fn generate_case_mapping(data: &UnicodeData) -> String {
    let mut file = String::new();

    file.push_str(HEADER.trim_start());
    file.push_str(&generate_tables("LOWER", &data.to_lower));
    file.push_str("\n\n");
    file.push_str(&generate_tables("UPPER", &data.to_upper));
    file
}

fn generate_tables(case: &str, data: &BTreeMap<u32, (u32, u32, u32)>) -> String {
    let (single, multi): (Vec<_>, Vec<_>) = data
        .iter()
        .map(to_mapping)
        .filter(|(k, _)| !k.0.is_ascii())
        .partition(|(_, [_, s, t])| s.0 == '\0' && t.0 == '\0');

    let mut tables = String::new();

    write!(
        tables,
        "static {}CASE_TABLE_SINGLE: &[(char, char)] = &[{}];",
        case,
        fmt_list(single.into_iter().map(|(k, [v, _, _])| (k, v)))
    )
    .unwrap();

    tables.push_str("\n\n");

    write!(
        tables,
        "static {}CASE_TABLE_MULTI: &[(char, [char; 3])] = &[{}];",
        case,
        fmt_list(multi)
    )
    .unwrap();

    tables
}

fn to_mapping((key, (a, b, c)): (&u32, &(u32, u32, u32))) -> (CharEscape, [CharEscape; 3]) {
    (
        CharEscape(std::char::from_u32(*key).unwrap()),
        [
            CharEscape(std::char::from_u32(*a).unwrap()),
            CharEscape(std::char::from_u32(*b).unwrap()),
            CharEscape(std::char::from_u32(*c).unwrap()),
        ],
    )
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
        match bsearch_case_tables(c, LOWERCASE_TABLE_SINGLE, LOWERCASE_TABLE_MULTI) {
            Some(replacement) => replacement,
            None => [c, '\0', '\0'],
        }
    }
}

pub fn to_upper(c: char) -> [char; 3] {
    if c.is_ascii() {
        [(c as u8).to_ascii_uppercase() as char, '\0', '\0']
    } else {
        match bsearch_case_tables(c, UPPERCASE_TABLE_SINGLE, UPPERCASE_TABLE_MULTI) {
            Some(replacement) => replacement,
            None => [c, '\0', '\0'],
        }
    }
}

fn bsearch_case_tables(
    c: char,
    single: &[(char, char)],
    multi: &[(char, [char; 3])],
) -> Option<[char; 3]> {
    match single.binary_search_by(|&(key, _)| key.cmp(&c)) {
        Ok(i) => Some([single[i].1, '\0', '\0']),
        Err(_) => multi.binary_search_by(|&(key, _)| key.cmp(&c)).map(|i| multi[i].1).ok(),
    }
}
";
