//! Tidy check to ensure that there are no unused pre-interned symbols.
//!
//! Symbols are defined in the `symbols!` macro call in `compiler/rusc_span/src/symbol.rs`.

use crate::walk::{filter_not_rust, walk_many};
use regex::Regex;
use std::{collections::HashMap, path::Path};

/// Symbols that are not used as `sym::*`, but should be explicitly ignored by
/// this check.
static IGNORELIST: &[&str] = &[
    // a
];

/// Symbols that are used, but not picked up by this check.
///
/// This can happen if a symbol is used in macro interpolation and the macro
/// is not explicitly handled below.
static ALLOWLIST: &[&str] = &[
    "Hasher",
    "StructuralEq",
    "StructuralPartialEq",
    "alu32",
    "avx",
    "avx512bw",
    "avx512f",
    "cr",
    "d32",
    "derive_const",
    "local",
    "neon",
    "position",
    "rust_begin_unwind",
    "rust_eh_catch_typeinfo",
    "rust_eh_personality",
    "rustc_dump_env_program_clauses",
    "rustc_dump_program_clauses",
    "sse",
    "vfp2",
    "width",
    "xer",
];

pub fn check(compiler: &Path, librustdoc: &Path, tools: &Path, bad: &mut bool) {
    // Find the `Symbols { ... }` block in `rustc_span::symbol`.
    let symbols_path = compiler.join("rustc_span/src/symbol.rs");
    let symbols_file_contents: String = std::fs::read_to_string(&symbols_path).unwrap();
    let symbols_block = {
        let e = || panic!("Couldn't find `Symbols` block in `{}`", symbols_path.display());

        let start_pat = "    Symbols {\n";
        let start = symbols_file_contents.find(start_pat).unwrap_or_else(e) + start_pat.len();

        let end_pat = "\n    }\n}";
        let relative_end = symbols_file_contents[start..].find(end_pat).unwrap_or_else(e);

        &symbols_file_contents[start..start + relative_end]
    };

    // Extract all the symbol identifiers from the block.
    let mut symbols: HashMap<&str, bool> = symbols_block
        .split(',')
        .filter_map(|item| {
            let item = item.trim();
            let ident = item.split_once(':').map_or(item, |(lhs, _)| lhs.trim());
            // Skip multi-line literals or empty strings (`sym::unstable_location_reason_default`)
            if ident.is_empty() || ident.contains(char::is_whitespace) {
                None
            } else {
                // Allow all assembly registries
                let used = ident.contains("reg");
                Some((ident, used))
            }
        })
        // Add special cases that are not in the `Symbols` block.
        .chain([("macro_rules", false), ("integer", true)])
        .collect();

    // Add the symbols from the allowlist.
    for symbol in ALLOWLIST {
        set(&mut symbols, symbol);
    }

    // Add the symbols used in `declare_features!` macro calls.
    find_features(&mut symbols, compiler);

    // Add the symbols used in `compiler/rustc_builtin_macros/src/lib.rs`.
    find_builtins(&mut symbols, compiler);

    // Find all the symbol identifiers in `rustc_span` users.
    let clippy = tools.join("clippy");
    let rustfmt = tools.join("rustfmt");
    let miri = tools.join("miri");
    let paths = &[compiler, librustdoc, &clippy, &rustfmt, &miri];
    find_sym(&mut symbols, paths);

    let mut unused = symbols
        .iter()
        .filter(|&(k, &v)| !v && !IGNORELIST.contains(k))
        .map(|(&k, _)| k)
        .collect::<Vec<_>>();
    if !unused.is_empty() {
        unused.sort_unstable();
        tidy_error!(
            bad,
            "found {} unused pre-interned symbols in `{}`:\n    {}",
            unused.len(),
            symbols_path.display(),
            unused.join("\n    "),
        );
    }
}

fn find_sym(symbols: &mut HashMap<&str, bool>, paths: &[&Path]) {
    let sym_re = Regex::new(r"\bsym::\w+").unwrap();
    walk_many(paths, |path, _| filter_not_rust(path), &mut |_entry, contents| {
        for m in sym_re.find_iter(contents) {
            // skip `sym::`
            let symbol = &contents[m.start() + 5..m.end()];
            set(symbols, symbol);
        }
    });
}

/// Finds the symbols used in `declare_features!` macro calls and adds them to
/// the `symbols` map.
fn find_features(symbols: &mut HashMap<&str, bool>, compiler: &Path) {
    let start = Regex::new(r"declare_features!\s?(\(|\[|\{)").unwrap();
    let end = Regex::new(r"(\)|\]|\});\n").unwrap();
    let non_word = Regex::new(r"\W").unwrap();

    let rustc_feature_src = compiler.join("rustc_feature/src");
    let feature_files = [
        rustc_feature_src.join("accepted.rs"),
        rustc_feature_src.join("active.rs"),
        rustc_feature_src.join("removed.rs"),
    ];
    for file in feature_files {
        let contents = std::fs::read_to_string(&file).unwrap();
        let features_blocks_lines =
            regex_blocks(&start, &end, &contents).map(|s| s.lines()).flatten();
        for line in features_blocks_lines {
            let line = line.trim_start();
            if !line.starts_with('(') {
                continue;
            }
            let split: Vec<&str> = line.split(',').collect();
            if split.len() < 4 {
                continue;
            }
            let symbol = split[1].trim();
            if symbol.is_empty() || non_word.is_match(symbol) {
                continue;
            }
            set(symbols, symbol);
        }
    }
}

/// Finds the symbols used in compiler/rustc_builtin_macros/src/lib.rs
/// and adds them to the `symbols` map.
fn find_builtins(symbols: &mut HashMap<&str, bool>, compiler: &Path) {
    let path = compiler.join("rustc_builtin_macros/src/lib.rs");
    let contents = std::fs::read_to_string(path).unwrap();

    let end = Regex::new(r"(\)|\]|\})\s*").unwrap();
    for mac in ["register_bang", "register_attr", "register_derive"] {
        let start = Regex::new(&format!(r"{mac}!\s?(\(|\[|\{{)")).unwrap();
        let macro_block = regex_blocks(&start, &end, &contents).next().unwrap();
        for line in macro_block.lines() {
            let Some((symbol, _)) = line.trim().split_once(':') else { continue };
            set(symbols, symbol);
        }
    }
}

fn regex_blocks<'a>(
    start: &'a Regex,
    end: &'a Regex,
    s: &'a str,
) -> impl Iterator<Item = &'a str> + 'a {
    start.find_iter(s).map(move |m| {
        let start_idx = m.end();
        let end_idx = end.find(&s[start_idx..]).unwrap().start();
        &s[start_idx..start_idx + end_idx]
    })
}

fn set(symbols: &mut HashMap<&str, bool>, symbol: &str) {
    match symbols.get_mut(symbol) {
        Some(used) => *used = true,
        None => panic!("Symbols map doesn't contain `sym::{}`", symbol),
    }
}
