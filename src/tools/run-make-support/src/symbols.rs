use std::collections::BTreeSet;
use std::path::Path;

use object::{self, Object, ObjectSymbol};

/// Given an [`object::File`], find the exported dynamic symbol names via
/// [`object::Object::exports`]. This does not distinguish between which section the symbols appear
/// in.
#[track_caller]
pub fn exported_dynamic_symbol_names<'file>(file: &'file object::File<'file>) -> Vec<&'file str> {
    file.exports()
        .unwrap()
        .into_iter()
        .filter_map(|sym| std::str::from_utf8(sym.name()).ok())
        .collect()
}

/// Check an object file's symbols for any matching **substrings**. That is, if an object file
/// contains a symbol named `hello_world`, it will be matched against a provided `substrings` of
/// `["hello", "bar"]`.
///
/// Returns `true` if **any** of the symbols found in the object file at `path` contain a
/// **substring** listed in `substrings`.
///
/// Panics if `path` is not a valid object file readable by the current user or if `path` cannot be
/// parsed as a recognized object file.
///
/// # Platform-specific behavior
///
/// On Windows MSVC, the binary (e.g. `main.exe`) does not contain the symbols, but in the separate
/// PDB file instead. Furthermore, you will need to use [`crate::llvm::llvm_pdbutil`] as `object`
/// crate does not handle PDB files.
#[track_caller]
pub fn object_contains_any_symbol_substring<P, S>(path: P, substrings: &[S]) -> bool
where
    P: AsRef<Path>,
    S: AsRef<str>,
{
    let path = path.as_ref();
    let blob = crate::fs::read(path);
    let obj = object::File::parse(&*blob)
        .unwrap_or_else(|e| panic!("failed to parse `{}`: {e}", path.display()));
    let substrings = substrings.iter().map(|s| s.as_ref()).collect::<Vec<_>>();
    for sym in obj.symbols() {
        for substring in &substrings {
            if sym.name_bytes().unwrap().windows(substring.len()).any(|x| x == substring.as_bytes())
            {
                return true;
            }
        }
    }
    false
}

/// Check an object file's symbols for any exact matches against those provided in
/// `candidate_symbols`.
///
/// Returns `true` if **any** of the symbols found in the object file at `path` contain an **exact
/// match** against those listed in `candidate_symbols`. Take care to account for (1) platform
/// differences and (2) calling convention and symbol decorations differences.
///
/// Panics if `path` is not a valid object file readable by the current user or if `path` cannot be
/// parsed as a recognized object file.
///
/// # Platform-specific behavior
///
/// See [`object_contains_any_symbol_substring`].
#[track_caller]
pub fn object_contains_any_symbol<P, S>(path: P, candidate_symbols: &[S]) -> bool
where
    P: AsRef<Path>,
    S: AsRef<str>,
{
    let path = path.as_ref();
    let blob = crate::fs::read(path);
    let obj = object::File::parse(&*blob)
        .unwrap_or_else(|e| panic!("failed to parse `{}`: {e}", path.display()));
    let candidate_symbols = candidate_symbols.iter().map(|s| s.as_ref()).collect::<Vec<_>>();
    for sym in obj.symbols() {
        for candidate_symbol in &candidate_symbols {
            if sym.name_bytes().unwrap() == candidate_symbol.as_bytes() {
                return true;
            }
        }
    }
    false
}

#[derive(Debug, PartialEq)]
pub enum ContainsAllSymbolSubstringsOutcome<'a> {
    Ok,
    MissingSymbolSubstrings(BTreeSet<&'a str>),
}

/// Check an object file's symbols for presence of all of provided **substrings**. That is, if an
/// object file contains symbols `["hello", "goodbye", "world"]`, it will be matched against a list
/// of `substrings` of `["he", "go"]`. In this case, `he` is a substring of `hello`, and `go` is a
/// substring of `goodbye`, so each of `substrings` was found.
///
/// Returns `true` if **all** `substrings` were present in the names of symbols for the given object
/// file (as substrings of symbol names).
///
/// Panics if `path` is not a valid object file readable by the current user or if `path` cannot be
/// parsed as a recognized object file.
///
/// # Platform-specific behavior
///
/// See [`object_contains_any_symbol_substring`].
#[track_caller]
pub fn object_contains_all_symbol_substring<'s, P, S>(
    path: P,
    substrings: &'s [S],
) -> ContainsAllSymbolSubstringsOutcome<'s>
where
    P: AsRef<Path>,
    S: AsRef<str>,
{
    let path = path.as_ref();
    let blob = crate::fs::read(path);
    let obj = object::File::parse(&*blob)
        .unwrap_or_else(|e| panic!("failed to parse `{}`: {e}", path.display()));
    let substrings = substrings.iter().map(|s| s.as_ref());
    let mut unmatched_symbol_substrings = BTreeSet::from_iter(substrings);
    unmatched_symbol_substrings.retain(|unmatched_symbol_substring| {
        for sym in obj.symbols() {
            if sym
                .name_bytes()
                .unwrap()
                .windows(unmatched_symbol_substring.len())
                .any(|x| x == unmatched_symbol_substring.as_bytes())
            {
                return false;
            }
        }

        true
    });

    if unmatched_symbol_substrings.is_empty() {
        ContainsAllSymbolSubstringsOutcome::Ok
    } else {
        ContainsAllSymbolSubstringsOutcome::MissingSymbolSubstrings(unmatched_symbol_substrings)
    }
}

#[derive(Debug, PartialEq)]
pub enum ContainsAllSymbolsOutcome<'a> {
    Ok,
    MissingSymbols(BTreeSet<&'a str>),
}

/// Check an object file contains all symbols provided in `candidate_symbols`.
///
/// Returns `true` if **all** of the symbols in `candidate_symbols` are found within the object file
/// at `path` by **exact match**. Take care to account for (1) platform differences and (2) calling
/// convention and symbol decorations differences.
///
/// Panics if `path` is not a valid object file readable by the current user or if `path` cannot be
/// parsed as a recognized object file.
///
/// # Platform-specific behavior
///
/// See [`object_contains_any_symbol_substring`].
#[track_caller]
pub fn object_contains_all_symbols<P, S>(
    path: P,
    candidate_symbols: &[S],
) -> ContainsAllSymbolsOutcome<'_>
where
    P: AsRef<Path>,
    S: AsRef<str>,
{
    let path = path.as_ref();
    let blob = crate::fs::read(path);
    let obj = object::File::parse(&*blob)
        .unwrap_or_else(|e| panic!("failed to parse `{}`: {e}", path.display()));
    let candidate_symbols = candidate_symbols.iter().map(|s| s.as_ref());
    let mut unmatched_symbols = BTreeSet::from_iter(candidate_symbols);
    unmatched_symbols.retain(|unmatched_symbol| {
        for sym in obj.symbols() {
            if sym.name_bytes().unwrap() == unmatched_symbol.as_bytes() {
                return false;
            }
        }

        true
    });

    if unmatched_symbols.is_empty() {
        ContainsAllSymbolsOutcome::Ok
    } else {
        ContainsAllSymbolsOutcome::MissingSymbols(unmatched_symbols)
    }
}
