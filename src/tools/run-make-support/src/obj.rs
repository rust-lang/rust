//! Helpers for working with types from the [`object`] crate.

use std::collections::BTreeSet;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use bstr::BStr;
pub use object::read::archive::ArchiveFile;
use object::{self, Object, ObjectSymbol, Symbol};
pub use object::{File as ObjFile, Result as ObjResult};

use crate::rfs;

/// Thin wrapper for owning data used by `object`, which may be either an archive or a single
/// object file.
pub struct BinFile {
    path: PathBuf,
    data: Vec<u8>,
}

impl BinFile {
    /// Read an archive or object file at `path`.
    pub fn read_path<P: Into<PathBuf>>(path: P) -> Self {
        let path = path.into();
        Self { data: rfs::read(&path), path }
    }

    /// Access the owned buffer as an archive file.
    pub fn parse_as_archive_file(&self) -> ObjResult<ArchiveFile<'_>> {
        ArchiveFile::parse(self.data.as_slice())
    }

    /// Access the owned buffer as an archive file.
    pub fn parse_as_obj_file(&self) -> ObjResult<ObjFile<'_>> {
        ObjFile::parse(self.data.as_slice())
    }

    /// If the file is an archive, run the callback for each object file. If it is an object
    /// file, the callback will run once.
    ///
    /// The callback receives the parsed object file and its name in the archive or on disk.
    pub fn for_each_object(&self, mut f: impl FnMut(ObjFile<'_>, &BStr)) {
        // Try as an archive first.
        let as_archive = self.parse_as_archive_file();
        if let Ok(archive) = as_archive {
            for member in archive.members() {
                let member = member.expect("failed to access member");
                let obj_data = member.data(self.data.as_slice()).expect("failed to access object");
                let obj = ObjFile::parse(obj_data).expect("failed to parse object");
                f(obj, BStr::new(member.name()));
            }

            return;
        }

        // Fall back to parsing as an object file.
        let as_obj = self.parse_as_obj_file();
        if let Ok(obj) = as_obj {
            let path_os = self.path.as_os_str();
            let path = cfg_select! {
                unix => path_os.as_bytes(),
                _ => path_os
                    .to_str()
                    .unwrap_or_else(|| panic!("non-utf-8 path on non-unix: {:?}", self.path))
                    .as_bytes(),
            };
            f(obj, BStr::new(path));
            return;
        }

        panic!(
            "failed to parse {:?} as either an archive or a object file: {:?}, {:?}",
            self.path,
            as_archive.unwrap_err(),
            as_obj.unwrap_err(),
        );
    }

    /// Do something with each symbol in an archive or object file.
    ///
    /// The callback receives:
    ///
    /// * The symbol.
    /// * The parsed object file that contains tye symbol.
    /// * The name of the object file in its archive or on disk.
    pub fn for_each_symbol(&self, mut f: impl FnMut(Symbol<'_, '_>, &ObjFile<'_>, &BStr)) {
        self.for_each_object(|obj, obj_path| {
            obj.symbols().for_each(|sym| f(sym, &obj, obj_path));
        });
    }
}

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
