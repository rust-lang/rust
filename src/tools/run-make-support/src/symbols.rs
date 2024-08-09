use std::path::Path;

use object::{self, Object, ObjectSymbol, Symbol, SymbolIterator};

/// Iterate through the symbols in an object file.
///
/// Uses a callback because `SymbolIterator` does not own its data.
///
/// Panics if `path` is not a valid object file readable by the current user.
pub fn with_symbol_iter<P, F, R>(path: P, func: F) -> R
where
    P: AsRef<Path>,
    F: FnOnce(&mut SymbolIterator<'_, '_>) -> R,
{
    let raw_bytes = crate::fs::read(path);
    let f = object::File::parse(raw_bytes.as_slice()).expect("unable to parse file");
    let mut iter = f.symbols();
    func(&mut iter)
}

/// Check an object file's symbols for substrings.
///
/// Returns `true` if any of the symbols found in the object file at
/// `path` contain a substring listed in `substrings`.
///
/// Panics if `path` is not a valid object file readable by the current user.
pub fn any_symbol_contains(path: impl AsRef<Path>, substrings: &[&str]) -> bool {
    with_symbol_iter(path, |syms| {
        for sym in syms {
            for substring in substrings {
                if sym
                    .name_bytes()
                    .unwrap()
                    .windows(substring.len())
                    .any(|x| x == substring.as_bytes())
                {
                    eprintln!("{:?} contains {}", sym, substring);
                    return true;
                }
            }
        }
        false
    })
}

/// Get a list of symbols that are in `symbol_names` but not the final binary.
///
/// The symbols must also match `pred`.
///
/// The symbol names must match exactly.
///
/// Panics if `path` is not a valid object file readable by the current user.
pub fn missing_exact_symbols<'a>(
    path: impl AsRef<Path>,
    symbol_names: &[&'a str],
    pred: impl Fn(&Symbol<'_, '_>) -> bool,
) -> Vec<&'a str> {
    let mut found = vec![false; symbol_names.len()];
    with_symbol_iter(path, |syms| {
        for sym in syms.filter(&pred) {
            for (i, symbol_name) in symbol_names.iter().enumerate() {
                found[i] |= sym.name_bytes().unwrap() == symbol_name.as_bytes();
            }
        }
    });
    return found
        .iter()
        .enumerate()
        .filter_map(|(i, found)| if !*found { Some(symbol_names[i]) } else { None })
        .collect();
}

/// Assert that the symbol file contains all of the listed symbols and they all match the given predicate
pub fn assert_contains_exact_symbols(
    path: impl AsRef<Path>,
    symbol_names: &[&str],
    pred: impl Fn(&Symbol<'_, '_>) -> bool,
) {
    let missing = missing_exact_symbols(path.as_ref(), symbol_names, pred);
    if missing.len() > 0 {
        eprintln!("{} does not contain symbol(s): ", path.as_ref().display());
        for sn in missing {
            eprintln!("* {}", sn);
        }
        panic!("missing symbols");
    }
}
