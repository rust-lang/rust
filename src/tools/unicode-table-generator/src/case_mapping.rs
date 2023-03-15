use crate::{fmt_list, UnicodeData};
use std::{collections::BTreeMap, fmt};

pub(crate) fn generate_case_mapping(data: &UnicodeData) -> String {
    let mut file = String::new();

    file.push_str(HEADER.trim_start());
    file.push_str(&generate_table("LOWER", &data.to_lower));
    file.push_str("\n\n");
    file.push_str(&generate_table("UPPER", &data.to_upper));
    file
}

fn generate_table(case: &str, data: &BTreeMap<u32, (u32, u32, u32)>) -> String {
    format!(
        "static {}CASE_TABLE: &[(char, [char; 3])] = &[{}];",
        case,
        fmt_list(data.iter().map(to_mapping).filter(|(k, _)| !k.0.is_ascii()))
    )
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
        match bsearch_case_table(c, LOWERCASE_TABLE) {
            None => [c, '\0', '\0'],
            Some(index) => LOWERCASE_TABLE[index].1,
        }
    }
}

pub fn to_upper(c: char) -> [char; 3] {
    if c.is_ascii() {
        [(c as u8).to_ascii_uppercase() as char, '\0', '\0']
    } else {
        match bsearch_case_table(c, UPPERCASE_TABLE) {
            None => [c, '\0', '\0'],
            Some(index) => UPPERCASE_TABLE[index].1,
        }
    }
}

fn bsearch_case_table(c: char, table: &[(char, [char; 3])]) -> Option<usize> {
    table.binary_search_by(|&(key, _)| key.cmp(&c)).ok()
}
";
