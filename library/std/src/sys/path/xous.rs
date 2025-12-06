use crate::ffi::OsStr;
use crate::io;
use crate::path::{Path, PathBuf, Prefix};

#[inline]
pub fn is_sep_byte(b: u8) -> bool {
    b == b'/' || b == b':'
}

#[inline]
pub fn is_verbatim_sep(b: u8) -> bool {
    b == b'/'
}

#[inline]
pub fn parse_prefix(_: &OsStr) -> Option<Prefix<'_>> {
    None
}

pub const HAS_PREFIXES: bool = false;
pub const MAIN_SEP_STR: &str = "/";
pub const MAIN_SEP: char = '/';
pub const ALLOWED_SEP: &[char] = &['/', ':'];

/// Make a POSIX path absolute without changing its semantics.
pub(crate) fn absolute(path: &Path) -> io::Result<PathBuf> {
    Ok(path.to_owned())
}

pub(crate) fn is_absolute(path: &Path) -> bool {
    path.has_root() && path.prefix().is_some()
}

#[derive(Debug, PartialEq)]
pub enum BasisSplitError {
    /// This basis couldn't be split because it's the root element
    Root,
    /// This basis couldn't be split because it's listing all bases
    All,
    /// A path component was empty
    EmptyComponent,
}

#[derive(Default, Debug, PartialEq)]
pub struct SplitPath<'a> {
    pub basis: Option<&'a str>,
    pub dict: Option<&'a str>,
    pub key: Option<&'a str>,
}

/// Split a path into its constituent Basis, Dict, and Key, if the path is legal.
/// A Basis of `None` indicates the default Basis.
pub fn split_basis_dict_key<'a>(src: &'a str) -> Result<SplitPath<'a>, BasisSplitError> {
    let mut current_string = src;
    let mut path = SplitPath::default();

    // Special case for root
    if src == ":" || src == "/" {
        return Err(BasisSplitError::Root);
    }
    // Special case for listing all bases
    if src == "::" || src == "//" || src == "/:" || src == ":/" {
        return Err(BasisSplitError::All);
    }

    // If the string is an absolute path, then the first component is a basis
    if let Some(src) = src.strip_prefix(ALLOWED_SEP) {
        // See if we can split off a second separator which will get us the end
        // of the Basis name
        if let Some((maybe_basis, remainder)) = src.split_once(ALLOWED_SEP) {
            // If the basis is empty, pull it from the default Basis function.
            // Otherwise, use the specified basis.
            if !maybe_basis.is_empty() {
                path.basis = Some(maybe_basis);
            }
            current_string = remainder;
        }
    }

    // Now that we have a fully-specified Basis, pull out the rest of the string.
    if let Some(components) = current_string.rsplit_once(ALLOWED_SEP) {
        let (mut dict, key) = components;
        if let Some(stripped_dict) = dict.strip_prefix(ALLOWED_SEP) {
            dict = stripped_dict;
        }
        // Don't allow empty dicts
        if components.0.is_empty() {
            return Err(BasisSplitError::EmptyComponent);
        }
        for path_component in dict.split(ALLOWED_SEP) {
            if path_component.is_empty() {
                return Err(BasisSplitError::EmptyComponent);
            }
        }

        path.dict = Some(dict);
        // Don't allow empty keys
        if !key.is_empty() {
            path.key = Some(key);
        }
    } else if !current_string.is_empty() {
        path.dict = Some(current_string)
    }

    Ok(path)
}

#[cfg(test)]
fn assert_basis_dict_key(test: &str, basis: Option<&str>, dict: Option<&str>, key: Option<&str>) {
    let path = split_basis_dict_key(test).unwrap();
    assert_eq!(path.basis, basis);
    assert_eq!(path.dict, dict);
    assert_eq!(path.key, key);
}

#[test]
fn basis() {
    // A basis named "Home Wifi"
    assert_basis_dict_key(":Home Wifi:", Some("Home Wifi"), None, None);

    // :.System: -- A basis named ".System"
    assert_basis_dict_key(":.System:", Some(".System"), None, None);
}

#[test]
fn dict() {
    // wlan.networks -- A dict named "wlan.networks" in the default basis
    assert_basis_dict_key("wlan.networks", None, Some("wlan.networks"), None);
}

#[test]
fn key() {
    // wlan.networks/recent -- A dict named "wlan.networks/recent", which may be considered a path, in the default basis. This also describes a key called "recent" in the dict "wlan.networks", depending on whether
    assert_basis_dict_key("wlan.networks/recent", None, Some("wlan.networks"), Some("recent"));
}

#[test]
fn key_legacy() {
    // wlan.networks/recent -- A dict named "wlan.networks:recent", which may be considered a path, in the default basis. This also describes a key called "recent" in the dict "wlan.networks", depending on whether
    assert_basis_dict_key("wlan.networks:recent", None, Some("wlan.networks"), Some("recent"));
}

#[test]
fn dict_in_key() {
    // :.System:wlan.networks -- A dict named "wlan.networks" in the basis ".System"
    assert_basis_dict_key(":.System:wlan.networks", Some(".System"), Some("wlan.networks"), None);
}

#[test]
fn basis_dict_key() {
    // :.System:wlan.networks/recent -- a fully-qualified path, describing a key "recent" in the dict "wlan.networks" in the basis ".System".
    assert_basis_dict_key(
        ":.System:wlan.networks/recent",
        Some(".System"),
        Some("wlan.networks"),
        Some("recent"),
    );
}

#[test]
fn basis_dict_key_legacy() {
    // :.System:wlan.networks:recent -- a fully-qualified path, describing a key "recent" in the dict "wlan.networks" in the basis ".System".
    assert_basis_dict_key(
        ":.System:wlan.networks:recent",
        Some(".System"),
        Some("wlan.networks"),
        Some("recent"),
    );
}

#[test]
fn root() {
    // : -- The root, which lists every basis. Files cannot be created here. "Directories" can be created and destroyed, which corresponds to creating and destroying bases.
    assert_eq!(split_basis_dict_key(":"), Err(BasisSplitError::Root));
    assert_eq!(split_basis_dict_key("/"), Err(BasisSplitError::Root));
}

#[test]
fn all_bases() {
    // :: -- An empty basis is a synonym for all bases, so this corresponds to listing all dicts in the root of the default basis.
    assert_eq!(split_basis_dict_key("::"), Err(BasisSplitError::All));
    assert_eq!(split_basis_dict_key("/:"), Err(BasisSplitError::All));
    assert_eq!(split_basis_dict_key(":/"), Err(BasisSplitError::All));
    assert_eq!(split_basis_dict_key("//"), Err(BasisSplitError::All));
}

#[test]
fn space_basis() {
    // : : -- A basis named " ". Legal, but questionable
    assert_basis_dict_key(": :", Some(" "), None, None);
}

#[test]
fn space_dict() {
    // -- A dict named " " in the default basis. Legal, but questionable.
    assert_basis_dict_key(" ", None, Some(" "), None);
}

#[test]
fn legacy_dict_key() {
    // : -- A key named " " in a dict called " ". Legal.
    assert_basis_dict_key(" : ", None, Some(" "), Some(" "));
}

#[test]
fn legacy_trailing_separator() {
    // baz: -- A dict named "baz" in the default basis with an extra ":" following. Legal.
    assert_basis_dict_key("baz:", None, Some("baz"), None);
}

#[test]
fn legacy_trailing_separator_two_dicts() {
    // baz:foo: -- Currently illegal, but may become equal to baz:foo in the future.
    assert_basis_dict_key("baz:foo:", None, Some("baz:foo"), None);
}

#[test]
fn empty_dict() {
    // ::: -- An key named ":" in an empty dict in the default basis. Illegal.
    assert_eq!(split_basis_dict_key(":::"), Err(BasisSplitError::EmptyComponent));
}

#[test]
fn empty_key_empty_dict() {
    // :::: -- An key named "::" in an empty dict in the default basis. Illegal.
    assert_eq!(split_basis_dict_key("::::"), Err(BasisSplitError::EmptyComponent));
}

#[test]
fn key_in_default_basis() {
    // ::foo -- A dict "foo" in the default basis.
    assert_basis_dict_key("::foo", None, Some("foo"), None);
}

#[test]
fn starts_with_slash() {
    assert_basis_dict_key("/test/path", Some("test"), Some("path"), None);
}

#[test]
fn non_legacy_basis() {
    assert_basis_dict_key(
        "/.System/wlan.networks/recent",
        Some(".System"),
        Some("wlan.networks"),
        Some("recent"),
    );
}
