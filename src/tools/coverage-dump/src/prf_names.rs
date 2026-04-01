use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;

use crate::llvm_utils::{truncated_md5, unescape_llvm_string_contents};
use crate::parser::Parser;

/// Scans through the contents of an LLVM IR assembly file to find `__llvm_prf_names`
/// entries, decodes them, and creates a table that maps name hash values to
/// (demangled) function names.
pub(crate) fn make_function_names_table(llvm_ir: &str) -> anyhow::Result<HashMap<u64, String>> {
    fn prf_names_payload(line: &str) -> Option<&str> {
        let re = {
            // We cheat a little bit and match the variable name `@__llvm_prf_nm`
            // rather than the section name, because the section name is harder
            // to extract and differs across Linux/Windows/macOS.
            static RE: OnceLock<Regex> = OnceLock::new();
            RE.get_or_init(|| {
                Regex::new(r#"^@__llvm_prf_nm =.*\[[0-9]+ x i8\] c"([^"]*)".*$"#).unwrap()
            })
        };

        let payload = re.captures(line)?.get(1).unwrap().as_str();
        Some(payload)
    }

    fn demangle_if_able(symbol_name_bytes: &[u8]) -> anyhow::Result<String> {
        // In practice, raw symbol names should always be ASCII.
        let symbol_name_str = std::str::from_utf8(symbol_name_bytes)?;
        match rustc_demangle::try_demangle(symbol_name_str) {
            Ok(d) => Ok(format!("{d:#}")),
            // If demangling failed, don't treat it as an error. This lets us
            // run the dump tool against non-Rust coverage maps produced by
            // `clang`, for testing purposes.
            Err(_) => Ok(format!("(couldn't demangle) {symbol_name_str}")),
        }
    }

    let mut map = HashMap::new();

    for payload in llvm_ir.lines().filter_map(prf_names_payload).map(unescape_llvm_string_contents)
    {
        let mut parser = Parser::new(&payload);
        let uncompressed_bytes = parser.read_chunk_to_uncompressed_bytes()?;
        parser.ensure_empty()?;

        // Symbol names in the payload are separated by `0x01` bytes.
        for raw_name in uncompressed_bytes.split(|&b| b == 0x01) {
            let hash = truncated_md5(raw_name);
            let demangled = demangle_if_able(raw_name)?;
            map.insert(hash, demangled);
        }
    }

    Ok(map)
}
