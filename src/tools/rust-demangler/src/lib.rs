use regex::Regex;
use rustc_demangle::demangle;
use std::str::Lines;

const REPLACE_COLONS: &str = "::";

pub fn create_disambiguator_re() -> Regex {
    Regex::new(r"\[[a-f0-9]{5,16}\]::").unwrap()
}

pub fn demangle_lines(lines: Lines<'_>, strip_crate_disambiguators: Option<Regex>) -> Vec<String> {
    let mut demangled_lines = Vec::new();
    for mangled in lines {
        let mut demangled = demangle(mangled).to_string();
        if let Some(re) = &strip_crate_disambiguators {
            demangled = re.replace_all(&demangled, REPLACE_COLONS).to_string();
        }
        demangled_lines.push(demangled);
    }
    demangled_lines
}
