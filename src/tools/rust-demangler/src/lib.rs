use regex::Regex;
use rustc_demangle::demangle;

const REPLACE_COLONS: &str = "::";

pub fn create_disambiguator_re() -> Regex {
    Regex::new(r"\[[a-f0-9]{5,16}\]::").unwrap()
}

pub fn demangle_lines(buffer: &str, strip_crate_disambiguators: Option<Regex>) -> Vec<String> {
    let lines = buffer.lines();
    let mut demangled_lines = Vec::new();
    for mangled in lines {
        let mut demangled = demangle(mangled).to_string();
        if let Some(re) = &strip_crate_disambiguators {
            demangled = re.replace_all(&demangled, REPLACE_COLONS).to_string();
        }
        demangled_lines.push(demangled);
    }
    demangled_lines.push("".to_string());
    demangled_lines
}
