use std::path::Path;

use object::{self, Object, ObjectSymbol, SymbolIterator};

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
    let path = path.as_ref();
    with_symbol_iter(path, |syms| {
        for sym in syms {
            for substring in substrings {
                if sym
                    .name_bytes()
                    .unwrap()
                    .windows(substring.len())
                    .any(|x| x == substring.as_bytes())
                {
                    eprintln!("{:?} contains {} in {}", sym, substring, path.display());
                    return true;
                }
            }
        }
        false
    })
}

/// Check if an object file contains *all* of the given symbols.
///
/// The symbol names must match exactly.
///
/// Panics if `path` is not a valid object file readable by the current user.
pub fn contains_exact_symbols(path: impl AsRef<Path>, symbol_names: &[&str]) -> bool {
    let mut found = vec![false; symbol_names.len()];
    with_symbol_iter(path, |syms| {
        for sym in syms {
            for (i, symbol_name) in symbol_names.iter().enumerate() {
                found[i] |= sym.name_bytes().unwrap() == symbol_name.as_bytes();
            }
        }
    });
    let result = found.iter().all(|x| *x);
    if !result {
        eprintln!("does not contain symbol(s): ");
        for i in 0..found.len() {
            if !found[i] {
                eprintln!("* {}", symbol_names[i]);
            }
        }
    }
    result
}

pub fn print_symbols(path: impl AsRef<Path>) {
    let path = path.as_ref();
    println!("symbols in {}:", path.display());
    with_symbol_iter(path, |syms| {
        syms.for_each(|sym| println!("  {}", &sym.name().unwrap()));
    });
}
