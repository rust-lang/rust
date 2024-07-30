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
