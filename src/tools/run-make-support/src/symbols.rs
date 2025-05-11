use std::path::Path;

use object::{self, Object, ObjectSymbol, SymbolIterator};

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

/// Iterate through the symbols in an object file. See [`object::Object::symbols`].
///
/// Panics if `path` is not a valid object file readable by the current user or if `path` cannot be
/// parsed as a recognized object file.
#[track_caller]
pub fn with_symbol_iter<P, F, R>(path: P, func: F) -> R
where
    P: AsRef<Path>,
    F: FnOnce(&mut SymbolIterator<'_, '_>) -> R,
{
    let path = path.as_ref();
    let blob = crate::fs::read(path);
    let f = object::File::parse(&*blob)
        .unwrap_or_else(|e| panic!("failed to parse `{}`: {e}", path.display()));
    let mut iter = f.symbols();
    func(&mut iter)
}

/// Check an object file's symbols for substrings.
///
/// Returns `true` if any of the symbols found in the object file at `path` contain a substring
/// listed in `substrings`.
///
/// Panics if `path` is not a valid object file readable by the current user or if `path` cannot be
/// parsed as a recognized object file.
#[track_caller]
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
