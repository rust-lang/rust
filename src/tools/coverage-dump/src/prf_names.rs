use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{anyhow, ensure};
use regex::Regex;

use crate::parser::{Parser, unescape_llvm_string_contents};

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

    /// LLVM's profiler/coverage metadata often uses an MD5 hash truncated to
    /// 64 bits as a way to associate data stored in different tables/sections.
    fn truncated_md5(bytes: &[u8]) -> u64 {
        use md5::{Digest, Md5};
        let mut hasher = Md5::new();
        hasher.update(bytes);
        let hash: [u8; 8] = hasher.finalize().as_slice()[..8].try_into().unwrap();
        // The truncated hash is explicitly little-endian, regardless of host
        // or target platform. (See `MD5Result::low` in LLVM's `MD5.h`.)
        u64::from_le_bytes(hash)
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
        let uncompressed_len = parser.read_uleb128_usize()?;
        let compressed_len = parser.read_uleb128_usize()?;

        let uncompressed_bytes_vec;
        let uncompressed_bytes: &[u8] = if compressed_len == 0 {
            // The symbol name bytes are uncompressed, so read them directly.
            parser.read_n_bytes(uncompressed_len)?
        } else {
            // The symbol name bytes are compressed, so read and decompress them.
            let compressed_bytes = parser.read_n_bytes(compressed_len)?;

            uncompressed_bytes_vec = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(
                compressed_bytes,
                uncompressed_len,
            )
            .map_err(|e| anyhow!("{e:?}"))?;
            ensure!(uncompressed_bytes_vec.len() == uncompressed_len);

            &uncompressed_bytes_vec
        };

        // Symbol names in the payload are separated by `0x01` bytes.
        for raw_name in uncompressed_bytes.split(|&b| b == 0x01) {
            let hash = truncated_md5(raw_name);
            let demangled = demangle_if_able(raw_name)?;
            map.insert(hash, demangled);
        }

        parser.ensure_empty()?;
    }

    Ok(map)
}
