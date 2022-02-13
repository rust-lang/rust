use super::*;

use crate::collections::hash_map::DefaultHasher;
use crate::collections::{BTreeSet, HashSet};
use crate::hash::Hasher;
use crate::rc::Rc;
use crate::sync::Arc;
use core::hint::black_box;

macro_rules! t (
    ($path:expr, iter: $iter:expr) => (
        {
            let path = Path::new($path);

            // Forward iteration
            let comps = path.iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect::<Vec<String>>();
            let exp: &[&str] = &$iter;
            let exps = exp.iter().map(|s| s.to_string()).collect::<Vec<String>>();
            assert!(comps == exps, "iter: Expected {:?}, found {:?}",
                    exps, comps);

            // Reverse iteration
            let comps = Path::new($path).iter().rev()
                .map(|p| p.to_string_lossy().into_owned())
                .collect::<Vec<String>>();
            let exps = exps.into_iter().rev().collect::<Vec<String>>();
            assert!(comps == exps, "iter().rev(): Expected {:?}, found {:?}",
                    exps, comps);
        }
    );

    ($path:expr, has_root: $has_root:expr, is_absolute: $is_absolute:expr) => (
        {
            let path = Path::new($path);

            let act_root = path.has_root();
            assert!(act_root == $has_root, "has_root: Expected {:?}, found {:?}",
                    $has_root, act_root);

            let act_abs = path.is_absolute();
            assert!(act_abs == $is_absolute, "is_absolute: Expected {:?}, found {:?}",
                    $is_absolute, act_abs);
        }
    );

    ($path:expr, parent: $parent:expr, file_name: $file:expr) => (
        {
            let path = Path::new($path);

            let parent = path.parent().map(|p| p.to_str().unwrap());
            let exp_parent: Option<&str> = $parent;
            assert!(parent == exp_parent, "parent: Expected {:?}, found {:?}",
                    exp_parent, parent);

            let file = path.file_name().map(|p| p.to_str().unwrap());
            let exp_file: Option<&str> = $file;
            assert!(file == exp_file, "file_name: Expected {:?}, found {:?}",
                    exp_file, file);
        }
    );

    ($path:expr, file_stem: $file_stem:expr, extension: $extension:expr) => (
        {
            let path = Path::new($path);

            let stem = path.file_stem().map(|p| p.to_str().unwrap());
            let exp_stem: Option<&str> = $file_stem;
            assert!(stem == exp_stem, "file_stem: Expected {:?}, found {:?}",
                    exp_stem, stem);

            let ext = path.extension().map(|p| p.to_str().unwrap());
            let exp_ext: Option<&str> = $extension;
            assert!(ext == exp_ext, "extension: Expected {:?}, found {:?}",
                    exp_ext, ext);
        }
    );

    ($path:expr, file_prefix: $file_prefix:expr, extension: $extension:expr) => (
        {
            let path = Path::new($path);

            let prefix = path.file_prefix().map(|p| p.to_str().unwrap());
            let exp_prefix: Option<&str> = $file_prefix;
            assert!(prefix == exp_prefix, "file_prefix: Expected {:?}, found {:?}",
                    exp_prefix, prefix);

            let ext = path.extension().map(|p| p.to_str().unwrap());
            let exp_ext: Option<&str> = $extension;
            assert!(ext == exp_ext, "extension: Expected {:?}, found {:?}",
                    exp_ext, ext);
        }
    );

    ($path:expr, iter: $iter:expr,
                 has_root: $has_root:expr, is_absolute: $is_absolute:expr,
                 parent: $parent:expr, file_name: $file:expr,
                 file_stem: $file_stem:expr, extension: $extension:expr,
                 file_prefix: $file_prefix:expr) => (
        {
            t!($path, iter: $iter);
            t!($path, has_root: $has_root, is_absolute: $is_absolute);
            t!($path, parent: $parent, file_name: $file);
            t!($path, file_stem: $file_stem, extension: $extension);
            t!($path, file_prefix: $file_prefix, extension: $extension);
        }
    );
);

#[test]
fn into() {
    use crate::borrow::Cow;

    let static_path = Path::new("/home/foo");
    let static_cow_path: Cow<'static, Path> = static_path.into();
    let pathbuf = PathBuf::from("/home/foo");

    {
        let path: &Path = &pathbuf;
        let borrowed_cow_path: Cow<'_, Path> = path.into();

        assert_eq!(static_cow_path, borrowed_cow_path);
    }

    let owned_cow_path: Cow<'static, Path> = pathbuf.into();

    assert_eq!(static_cow_path, owned_cow_path);
}

#[test]
#[cfg(unix)]
pub fn test_decompositions_unix() {
    t!("",
    iter: [],
    has_root: false,
    is_absolute: false,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("foo",
    iter: ["foo"],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None,
    file_prefix: Some("foo")
    );

    t!("/",
    iter: ["/"],
    has_root: true,
    is_absolute: true,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("/foo",
    iter: ["/", "foo"],
    has_root: true,
    is_absolute: true,
    parent: Some("/"),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None,
    file_prefix: Some("foo")
    );

    t!("foo/",
    iter: ["foo"],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None,
    file_prefix: Some("foo")
    );

    t!("/foo/",
    iter: ["/", "foo"],
    has_root: true,
    is_absolute: true,
    parent: Some("/"),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None,
    file_prefix: Some("foo")
    );

    t!("foo/bar",
    iter: ["foo", "bar"],
    has_root: false,
    is_absolute: false,
    parent: Some("foo"),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None,
    file_prefix: Some("bar")
    );

    t!("/foo/bar",
    iter: ["/", "foo", "bar"],
    has_root: true,
    is_absolute: true,
    parent: Some("/foo"),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None,
    file_prefix: Some("bar")
    );

    t!("///foo///",
    iter: ["/", "foo"],
    has_root: true,
    is_absolute: true,
    parent: Some("/"),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None,
    file_prefix: Some("foo")
    );

    t!("///foo///bar",
    iter: ["/", "foo", "bar"],
    has_root: true,
    is_absolute: true,
    parent: Some("///foo"),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None,
    file_prefix: Some("bar")
    );

    t!("./.",
    iter: ["."],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("/..",
    iter: ["/", ".."],
    has_root: true,
    is_absolute: true,
    parent: Some("/"),
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("../",
    iter: [".."],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("foo/.",
    iter: ["foo"],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None,
    file_prefix: Some("foo")
    );

    t!("foo/..",
    iter: ["foo", ".."],
    has_root: false,
    is_absolute: false,
    parent: Some("foo"),
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("foo/./",
    iter: ["foo"],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None,
    file_prefix: Some("foo")
    );

    t!("foo/./bar",
    iter: ["foo", "bar"],
    has_root: false,
    is_absolute: false,
    parent: Some("foo"),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None,
    file_prefix: Some("bar")
    );

    t!("foo/../",
    iter: ["foo", ".."],
    has_root: false,
    is_absolute: false,
    parent: Some("foo"),
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("foo/../bar",
    iter: ["foo", "..", "bar"],
    has_root: false,
    is_absolute: false,
    parent: Some("foo/.."),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None,
    file_prefix: Some("bar")
    );

    t!("./a",
    iter: [".", "a"],
    has_root: false,
    is_absolute: false,
    parent: Some("."),
    file_name: Some("a"),
    file_stem: Some("a"),
    extension: None,
    file_prefix: Some("a")
    );

    t!(".",
    iter: ["."],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("./",
    iter: ["."],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("a/b",
    iter: ["a", "b"],
    has_root: false,
    is_absolute: false,
    parent: Some("a"),
    file_name: Some("b"),
    file_stem: Some("b"),
    extension: None,
    file_prefix: Some("b")
    );

    t!("a//b",
    iter: ["a", "b"],
    has_root: false,
    is_absolute: false,
    parent: Some("a"),
    file_name: Some("b"),
    file_stem: Some("b"),
    extension: None,
    file_prefix: Some("b")
    );

    t!("a/./b",
    iter: ["a", "b"],
    has_root: false,
    is_absolute: false,
    parent: Some("a"),
    file_name: Some("b"),
    file_stem: Some("b"),
    extension: None,
    file_prefix: Some("b")
    );

    t!("a/b/c",
    iter: ["a", "b", "c"],
    has_root: false,
    is_absolute: false,
    parent: Some("a/b"),
    file_name: Some("c"),
    file_stem: Some("c"),
    extension: None,
    file_prefix: Some("c")
    );

    t!(".foo",
    iter: [".foo"],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: Some(".foo"),
    file_stem: Some(".foo"),
    extension: None,
    file_prefix: Some(".foo")
    );

    t!("a/.foo",
    iter: ["a", ".foo"],
    has_root: false,
    is_absolute: false,
    parent: Some("a"),
    file_name: Some(".foo"),
    file_stem: Some(".foo"),
    extension: None,
    file_prefix: Some(".foo")
    );

    t!("a/.rustfmt.toml",
    iter: ["a", ".rustfmt.toml"],
    has_root: false,
    is_absolute: false,
    parent: Some("a"),
    file_name: Some(".rustfmt.toml"),
    file_stem: Some(".rustfmt"),
    extension: Some("toml"),
    file_prefix: Some(".rustfmt")
    );

    t!("a/.x.y.z",
    iter: ["a", ".x.y.z"],
    has_root: false,
    is_absolute: false,
    parent: Some("a"),
    file_name: Some(".x.y.z"),
    file_stem: Some(".x.y"),
    extension: Some("z"),
    file_prefix: Some(".x")
    );
}

#[test]
#[cfg(windows)]
pub fn test_decompositions_windows() {
    t!("",
    iter: [],
    has_root: false,
    is_absolute: false,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("foo",
    iter: ["foo"],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None,
    file_prefix: Some("foo")
    );

    t!("/",
    iter: ["\\"],
    has_root: true,
    is_absolute: false,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("\\",
    iter: ["\\"],
    has_root: true,
    is_absolute: false,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("c:",
    iter: ["c:"],
    has_root: false,
    is_absolute: false,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("c:\\",
    iter: ["c:", "\\"],
    has_root: true,
    is_absolute: true,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("c:/",
    iter: ["c:", "\\"],
    has_root: true,
    is_absolute: true,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("/foo",
    iter: ["\\", "foo"],
    has_root: true,
    is_absolute: false,
    parent: Some("/"),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None,
    file_prefix: Some("foo")
    );

    t!("foo/",
    iter: ["foo"],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None,
    file_prefix: Some("foo")
    );

    t!("/foo/",
    iter: ["\\", "foo"],
    has_root: true,
    is_absolute: false,
    parent: Some("/"),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None,
    file_prefix: Some("foo")
    );

    t!("foo/bar",
    iter: ["foo", "bar"],
    has_root: false,
    is_absolute: false,
    parent: Some("foo"),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None,
    file_prefix: Some("bar")
    );

    t!("/foo/bar",
    iter: ["\\", "foo", "bar"],
    has_root: true,
    is_absolute: false,
    parent: Some("/foo"),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None,
    file_prefix: Some("bar")
    );

    t!("///foo///",
    iter: ["\\", "foo"],
    has_root: true,
    is_absolute: false,
    parent: Some("/"),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None,
    file_prefix: Some("foo")
    );

    t!("///foo///bar",
    iter: ["\\", "foo", "bar"],
    has_root: true,
    is_absolute: false,
    parent: Some("///foo"),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None,
    file_prefix: Some("bar")
    );

    t!("./.",
    iter: ["."],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("/..",
    iter: ["\\", ".."],
    has_root: true,
    is_absolute: false,
    parent: Some("/"),
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("../",
    iter: [".."],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("foo/.",
    iter: ["foo"],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None,
    file_prefix: Some("foo")
    );

    t!("foo/..",
    iter: ["foo", ".."],
    has_root: false,
    is_absolute: false,
    parent: Some("foo"),
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("foo/./",
    iter: ["foo"],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None,
    file_prefix: Some("foo")
    );

    t!("foo/./bar",
    iter: ["foo", "bar"],
    has_root: false,
    is_absolute: false,
    parent: Some("foo"),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None,
    file_prefix: Some("bar")
    );

    t!("foo/../",
    iter: ["foo", ".."],
    has_root: false,
    is_absolute: false,
    parent: Some("foo"),
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("foo/../bar",
    iter: ["foo", "..", "bar"],
    has_root: false,
    is_absolute: false,
    parent: Some("foo/.."),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None,
    file_prefix: Some("bar")
    );

    t!("./a",
    iter: [".", "a"],
    has_root: false,
    is_absolute: false,
    parent: Some("."),
    file_name: Some("a"),
    file_stem: Some("a"),
    extension: None,
    file_prefix: Some("a")
    );

    t!(".",
    iter: ["."],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("./",
    iter: ["."],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("a/b",
    iter: ["a", "b"],
    has_root: false,
    is_absolute: false,
    parent: Some("a"),
    file_name: Some("b"),
    file_stem: Some("b"),
    extension: None,
    file_prefix: Some("b")
    );

    t!("a//b",
    iter: ["a", "b"],
    has_root: false,
    is_absolute: false,
    parent: Some("a"),
    file_name: Some("b"),
    file_stem: Some("b"),
    extension: None,
    file_prefix: Some("b")
    );

    t!("a/./b",
    iter: ["a", "b"],
    has_root: false,
    is_absolute: false,
    parent: Some("a"),
    file_name: Some("b"),
    file_stem: Some("b"),
    extension: None,
    file_prefix: Some("b")
    );

    t!("a/b/c",
       iter: ["a", "b", "c"],
       has_root: false,
       is_absolute: false,
       parent: Some("a/b"),
       file_name: Some("c"),
       file_stem: Some("c"),
       extension: None,
       file_prefix: Some("c")
    );

    t!("a\\b\\c",
    iter: ["a", "b", "c"],
    has_root: false,
    is_absolute: false,
    parent: Some("a\\b"),
    file_name: Some("c"),
    file_stem: Some("c"),
    extension: None,
    file_prefix: Some("c")
    );

    t!("\\a",
    iter: ["\\", "a"],
    has_root: true,
    is_absolute: false,
    parent: Some("\\"),
    file_name: Some("a"),
    file_stem: Some("a"),
    extension: None,
    file_prefix: Some("a")
    );

    t!("c:\\foo.txt",
    iter: ["c:", "\\", "foo.txt"],
    has_root: true,
    is_absolute: true,
    parent: Some("c:\\"),
    file_name: Some("foo.txt"),
    file_stem: Some("foo"),
    extension: Some("txt"),
    file_prefix: Some("foo")
    );

    t!("\\\\server\\share\\foo.txt",
    iter: ["\\\\server\\share", "\\", "foo.txt"],
    has_root: true,
    is_absolute: true,
    parent: Some("\\\\server\\share\\"),
    file_name: Some("foo.txt"),
    file_stem: Some("foo"),
    extension: Some("txt"),
    file_prefix: Some("foo")
    );

    t!("\\\\server\\share",
    iter: ["\\\\server\\share", "\\"],
    has_root: true,
    is_absolute: true,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("\\\\server",
    iter: ["\\", "server"],
    has_root: true,
    is_absolute: false,
    parent: Some("\\"),
    file_name: Some("server"),
    file_stem: Some("server"),
    extension: None,
    file_prefix: Some("server")
    );

    t!("\\\\?\\bar\\foo.txt",
    iter: ["\\\\?\\bar", "\\", "foo.txt"],
    has_root: true,
    is_absolute: true,
    parent: Some("\\\\?\\bar\\"),
    file_name: Some("foo.txt"),
    file_stem: Some("foo"),
    extension: Some("txt"),
    file_prefix: Some("foo")
    );

    t!("\\\\?\\bar",
    iter: ["\\\\?\\bar"],
    has_root: true,
    is_absolute: true,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("\\\\?\\",
    iter: ["\\\\?\\"],
    has_root: true,
    is_absolute: true,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("\\\\?\\UNC\\server\\share\\foo.txt",
    iter: ["\\\\?\\UNC\\server\\share", "\\", "foo.txt"],
    has_root: true,
    is_absolute: true,
    parent: Some("\\\\?\\UNC\\server\\share\\"),
    file_name: Some("foo.txt"),
    file_stem: Some("foo"),
    extension: Some("txt"),
    file_prefix: Some("foo")
    );

    t!("\\\\?\\UNC\\server",
    iter: ["\\\\?\\UNC\\server"],
    has_root: true,
    is_absolute: true,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("\\\\?\\UNC\\",
    iter: ["\\\\?\\UNC\\"],
    has_root: true,
    is_absolute: true,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("\\\\?\\C:\\foo.txt",
    iter: ["\\\\?\\C:", "\\", "foo.txt"],
    has_root: true,
    is_absolute: true,
    parent: Some("\\\\?\\C:\\"),
    file_name: Some("foo.txt"),
    file_stem: Some("foo"),
    extension: Some("txt"),
    file_prefix: Some("foo")
    );

    t!("\\\\?\\C:\\",
    iter: ["\\\\?\\C:", "\\"],
    has_root: true,
    is_absolute: true,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("\\\\?\\C:",
    iter: ["\\\\?\\C:"],
    has_root: true,
    is_absolute: true,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("\\\\?\\foo/bar",
    iter: ["\\\\?\\foo/bar"],
    has_root: true,
    is_absolute: true,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("\\\\?\\C:/foo",
    iter: ["\\\\?\\C:/foo"],
    has_root: true,
    is_absolute: true,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("\\\\.\\foo\\bar",
    iter: ["\\\\.\\foo", "\\", "bar"],
    has_root: true,
    is_absolute: true,
    parent: Some("\\\\.\\foo\\"),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None,
    file_prefix: Some("bar")
    );

    t!("\\\\.\\foo",
    iter: ["\\\\.\\foo", "\\"],
    has_root: true,
    is_absolute: true,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("\\\\.\\foo/bar",
    iter: ["\\\\.\\foo", "\\", "bar"],
    has_root: true,
    is_absolute: true,
    parent: Some("\\\\.\\foo/"),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None,
    file_prefix: Some("bar")
    );

    t!("\\\\.\\foo\\bar/baz",
    iter: ["\\\\.\\foo", "\\", "bar", "baz"],
    has_root: true,
    is_absolute: true,
    parent: Some("\\\\.\\foo\\bar"),
    file_name: Some("baz"),
    file_stem: Some("baz"),
    extension: None,
    file_prefix: Some("baz")
    );

    t!("\\\\.\\",
    iter: ["\\\\.\\", "\\"],
    has_root: true,
    is_absolute: true,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None,
    file_prefix: None
    );

    t!("\\\\?\\a\\b\\",
    iter: ["\\\\?\\a", "\\", "b"],
    has_root: true,
    is_absolute: true,
    parent: Some("\\\\?\\a\\"),
    file_name: Some("b"),
    file_stem: Some("b"),
    extension: None,
    file_prefix: Some("b")
    );

    t!("\\\\?\\C:\\foo.txt.zip",
    iter: ["\\\\?\\C:", "\\", "foo.txt.zip"],
    has_root: true,
    is_absolute: true,
    parent: Some("\\\\?\\C:\\"),
    file_name: Some("foo.txt.zip"),
    file_stem: Some("foo.txt"),
    extension: Some("zip"),
    file_prefix: Some("foo")
    );

    t!("\\\\?\\C:\\.foo.txt.zip",
    iter: ["\\\\?\\C:", "\\", ".foo.txt.zip"],
    has_root: true,
    is_absolute: true,
    parent: Some("\\\\?\\C:\\"),
    file_name: Some(".foo.txt.zip"),
    file_stem: Some(".foo.txt"),
    extension: Some("zip"),
    file_prefix: Some(".foo")
    );

    t!("\\\\?\\C:\\.foo",
    iter: ["\\\\?\\C:", "\\", ".foo"],
    has_root: true,
    is_absolute: true,
    parent: Some("\\\\?\\C:\\"),
    file_name: Some(".foo"),
    file_stem: Some(".foo"),
    extension: None,
    file_prefix: Some(".foo")
    );

    t!("a/.x.y.z",
    iter: ["a", ".x.y.z"],
    has_root: false,
    is_absolute: false,
    parent: Some("a"),
    file_name: Some(".x.y.z"),
    file_stem: Some(".x.y"),
    extension: Some("z"),
    file_prefix: Some(".x")
    );
}

#[test]
pub fn test_stem_ext() {
    t!("foo",
    file_stem: Some("foo"),
    extension: None
    );

    t!("foo.",
    file_stem: Some("foo"),
    extension: Some("")
    );

    t!(".foo",
    file_stem: Some(".foo"),
    extension: None
    );

    t!("foo.txt",
    file_stem: Some("foo"),
    extension: Some("txt")
    );

    t!("foo.bar.txt",
    file_stem: Some("foo.bar"),
    extension: Some("txt")
    );

    t!("foo.bar.",
    file_stem: Some("foo.bar"),
    extension: Some("")
    );

    t!(".", file_stem: None, extension: None);

    t!("..", file_stem: None, extension: None);

    t!(".x.y.z", file_stem: Some(".x.y"), extension: Some("z"));

    t!("..x.y.z", file_stem: Some("..x.y"), extension: Some("z"));

    t!("", file_stem: None, extension: None);
}

#[test]
pub fn test_prefix_ext() {
    t!("foo",
    file_prefix: Some("foo"),
    extension: None
    );

    t!("foo.",
    file_prefix: Some("foo"),
    extension: Some("")
    );

    t!(".foo",
    file_prefix: Some(".foo"),
    extension: None
    );

    t!("foo.txt",
    file_prefix: Some("foo"),
    extension: Some("txt")
    );

    t!("foo.bar.txt",
    file_prefix: Some("foo"),
    extension: Some("txt")
    );

    t!("foo.bar.",
    file_prefix: Some("foo"),
    extension: Some("")
    );

    t!(".", file_prefix: None, extension: None);

    t!("..", file_prefix: None, extension: None);

    t!(".x.y.z", file_prefix: Some(".x"), extension: Some("z"));

    t!("..x.y.z", file_prefix: Some("."), extension: Some("z"));

    t!("", file_prefix: None, extension: None);
}

#[test]
pub fn test_push() {
    macro_rules! tp (
        ($path:expr, $push:expr, $expected:expr) => ( {
            let mut actual = PathBuf::from($path);
            actual.push($push);
            assert!(actual.to_str() == Some($expected),
                    "pushing {:?} onto {:?}: Expected {:?}, got {:?}",
                    $push, $path, $expected, actual.to_str().unwrap());
        });
    );

    if cfg!(unix) || cfg!(all(target_env = "sgx", target_vendor = "fortanix")) {
        tp!("", "foo", "foo");
        tp!("foo", "bar", "foo/bar");
        tp!("foo/", "bar", "foo/bar");
        tp!("foo//", "bar", "foo//bar");
        tp!("foo/.", "bar", "foo/./bar");
        tp!("foo./.", "bar", "foo././bar");
        tp!("foo", "", "foo/");
        tp!("foo", ".", "foo/.");
        tp!("foo", "..", "foo/..");
        tp!("foo", "/", "/");
        tp!("/foo/bar", "/", "/");
        tp!("/foo/bar", "/baz", "/baz");
        tp!("/foo/bar", "./baz", "/foo/bar/./baz");
    } else {
        tp!("", "foo", "foo");
        tp!("foo", "bar", r"foo\bar");
        tp!("foo/", "bar", r"foo/bar");
        tp!(r"foo\", "bar", r"foo\bar");
        tp!("foo//", "bar", r"foo//bar");
        tp!(r"foo\\", "bar", r"foo\\bar");
        tp!("foo/.", "bar", r"foo/.\bar");
        tp!("foo./.", "bar", r"foo./.\bar");
        tp!(r"foo\.", "bar", r"foo\.\bar");
        tp!(r"foo.\.", "bar", r"foo.\.\bar");
        tp!("foo", "", "foo\\");
        tp!("foo", ".", r"foo\.");
        tp!("foo", "..", r"foo\..");
        tp!("foo", "/", "/");
        tp!("foo", r"\", r"\");
        tp!("/foo/bar", "/", "/");
        tp!(r"\foo\bar", r"\", r"\");
        tp!("/foo/bar", "/baz", "/baz");
        tp!("/foo/bar", r"\baz", r"\baz");
        tp!("/foo/bar", "./baz", r"/foo/bar\./baz");
        tp!("/foo/bar", r".\baz", r"/foo/bar\.\baz");

        tp!("c:\\", "windows", "c:\\windows");
        tp!("c:", "windows", "c:windows");

        tp!("a\\b\\c", "d", "a\\b\\c\\d");
        tp!("\\a\\b\\c", "d", "\\a\\b\\c\\d");
        tp!("a\\b", "c\\d", "a\\b\\c\\d");
        tp!("a\\b", "\\c\\d", "\\c\\d");
        tp!("a\\b", ".", "a\\b\\.");
        tp!("a\\b", "..\\c", "a\\b\\..\\c");
        tp!("a\\b", "C:a.txt", "C:a.txt");
        tp!("a\\b", "C:\\a.txt", "C:\\a.txt");
        tp!("C:\\a", "C:\\b.txt", "C:\\b.txt");
        tp!("C:\\a\\b\\c", "C:d", "C:d");
        tp!("C:a\\b\\c", "C:d", "C:d");
        tp!("C:", r"a\b\c", r"C:a\b\c");
        tp!("C:", r"..\a", r"C:..\a");
        tp!("\\\\server\\share\\foo", "bar", "\\\\server\\share\\foo\\bar");
        tp!("\\\\server\\share\\foo", "C:baz", "C:baz");
        tp!("\\\\?\\C:\\a\\b", "C:c\\d", "C:c\\d");
        tp!("\\\\?\\C:a\\b", "C:c\\d", "C:c\\d");
        tp!("\\\\?\\C:\\a\\b", "C:\\c\\d", "C:\\c\\d");
        tp!("\\\\?\\foo\\bar", "baz", "\\\\?\\foo\\bar\\baz");
        tp!("\\\\?\\UNC\\server\\share\\foo", "bar", "\\\\?\\UNC\\server\\share\\foo\\bar");
        tp!("\\\\?\\UNC\\server\\share", "C:\\a", "C:\\a");
        tp!("\\\\?\\UNC\\server\\share", "C:a", "C:a");

        // Note: modified from old path API
        tp!("\\\\?\\UNC\\server", "foo", "\\\\?\\UNC\\server\\foo");

        tp!("C:\\a", "\\\\?\\UNC\\server\\share", "\\\\?\\UNC\\server\\share");
        tp!("\\\\.\\foo\\bar", "baz", "\\\\.\\foo\\bar\\baz");
        tp!("\\\\.\\foo\\bar", "C:a", "C:a");
        // again, not sure about the following, but I'm assuming \\.\ should be verbatim
        tp!("\\\\.\\foo", "..\\bar", "\\\\.\\foo\\..\\bar");

        tp!("\\\\?\\C:", "foo", "\\\\?\\C:\\foo"); // this is a weird one

        tp!(r"\\?\C:\bar", "../foo", r"\\?\C:\foo");
        tp!(r"\\?\C:\bar", "../../foo", r"\\?\C:\foo");
        tp!(r"\\?\C:\", "../foo", r"\\?\C:\foo");
        tp!(r"\\?\C:", r"D:\foo/./", r"D:\foo/./");
        tp!(r"\\?\C:", r"\\?\D:\foo\.\", r"\\?\D:\foo\.\");
        tp!(r"\\?\A:\x\y", "/foo", r"\\?\A:\foo");
        tp!(r"\\?\A:", r"..\foo\.", r"\\?\A:\foo");
        tp!(r"\\?\A:\x\y", r".\foo\.", r"\\?\A:\x\y\foo");
        tp!(r"\\?\A:\x\y", r"", r"\\?\A:\x\y\");
    }
}

#[test]
pub fn test_pop() {
    macro_rules! tp (
        ($path:expr, $expected:expr, $output:expr) => ( {
            let mut actual = PathBuf::from($path);
            let output = actual.pop();
            assert!(actual.to_str() == Some($expected) && output == $output,
                    "popping from {:?}: Expected {:?}/{:?}, got {:?}/{:?}",
                    $path, $expected, $output,
                    actual.to_str().unwrap(), output);
        });
    );

    tp!("", "", false);
    tp!("/", "/", false);
    tp!("foo", "", true);
    tp!(".", "", true);
    tp!("/foo", "/", true);
    tp!("/foo/bar", "/foo", true);
    tp!("foo/bar", "foo", true);
    tp!("foo/.", "", true);
    tp!("foo//bar", "foo", true);

    if cfg!(windows) {
        tp!("a\\b\\c", "a\\b", true);
        tp!("\\a", "\\", true);
        tp!("\\", "\\", false);

        tp!("C:\\a\\b", "C:\\a", true);
        tp!("C:\\a", "C:\\", true);
        tp!("C:\\", "C:\\", false);
        tp!("C:a\\b", "C:a", true);
        tp!("C:a", "C:", true);
        tp!("C:", "C:", false);
        tp!("\\\\server\\share\\a\\b", "\\\\server\\share\\a", true);
        tp!("\\\\server\\share\\a", "\\\\server\\share\\", true);
        tp!("\\\\server\\share", "\\\\server\\share", false);
        tp!("\\\\?\\a\\b\\c", "\\\\?\\a\\b", true);
        tp!("\\\\?\\a\\b", "\\\\?\\a\\", true);
        tp!("\\\\?\\a", "\\\\?\\a", false);
        tp!("\\\\?\\C:\\a\\b", "\\\\?\\C:\\a", true);
        tp!("\\\\?\\C:\\a", "\\\\?\\C:\\", true);
        tp!("\\\\?\\C:\\", "\\\\?\\C:\\", false);
        tp!("\\\\?\\UNC\\server\\share\\a\\b", "\\\\?\\UNC\\server\\share\\a", true);
        tp!("\\\\?\\UNC\\server\\share\\a", "\\\\?\\UNC\\server\\share\\", true);
        tp!("\\\\?\\UNC\\server\\share", "\\\\?\\UNC\\server\\share", false);
        tp!("\\\\.\\a\\b\\c", "\\\\.\\a\\b", true);
        tp!("\\\\.\\a\\b", "\\\\.\\a\\", true);
        tp!("\\\\.\\a", "\\\\.\\a", false);

        tp!("\\\\?\\a\\b\\", "\\\\?\\a\\", true);
    }
}

#[test]
pub fn test_set_file_name() {
    macro_rules! tfn (
            ($path:expr, $file:expr, $expected:expr) => ( {
            let mut p = PathBuf::from($path);
            p.set_file_name($file);
            assert!(p.to_str() == Some($expected),
                    "setting file name of {:?} to {:?}: Expected {:?}, got {:?}",
                    $path, $file, $expected,
                    p.to_str().unwrap());
        });
    );

    tfn!("foo", "foo", "foo");
    tfn!("foo", "bar", "bar");
    tfn!("foo", "", "");
    tfn!("", "foo", "foo");
    if cfg!(unix) || cfg!(all(target_env = "sgx", target_vendor = "fortanix")) {
        tfn!(".", "foo", "./foo");
        tfn!("foo/", "bar", "bar");
        tfn!("foo/.", "bar", "bar");
        tfn!("..", "foo", "../foo");
        tfn!("foo/..", "bar", "foo/../bar");
        tfn!("/", "foo", "/foo");
    } else {
        tfn!(".", "foo", r".\foo");
        tfn!(r"foo\", "bar", r"bar");
        tfn!(r"foo\.", "bar", r"bar");
        tfn!("..", "foo", r"..\foo");
        tfn!(r"foo\..", "bar", r"foo\..\bar");
        tfn!(r"\", "foo", r"\foo");
    }
}

#[test]
pub fn test_set_extension() {
    macro_rules! tfe (
            ($path:expr, $ext:expr, $expected:expr, $output:expr) => ( {
            let mut p = PathBuf::from($path);
            let output = p.set_extension($ext);
            assert!(p.to_str() == Some($expected) && output == $output,
                    "setting extension of {:?} to {:?}: Expected {:?}/{:?}, got {:?}/{:?}",
                    $path, $ext, $expected, $output,
                    p.to_str().unwrap(), output);
        });
    );

    tfe!("foo", "txt", "foo.txt", true);
    tfe!("foo.bar", "txt", "foo.txt", true);
    tfe!("foo.bar.baz", "txt", "foo.bar.txt", true);
    tfe!(".test", "txt", ".test.txt", true);
    tfe!("foo.txt", "", "foo", true);
    tfe!("foo", "", "foo", true);
    tfe!("", "foo", "", false);
    tfe!(".", "foo", ".", false);
    tfe!("foo/", "bar", "foo.bar", true);
    tfe!("foo/.", "bar", "foo.bar", true);
    tfe!("..", "foo", "..", false);
    tfe!("foo/..", "bar", "foo/..", false);
    tfe!("/", "foo", "/", false);
}

#[test]
fn test_eq_receivers() {
    use crate::borrow::Cow;

    let borrowed: &Path = Path::new("foo/bar");
    let mut owned: PathBuf = PathBuf::new();
    owned.push("foo");
    owned.push("bar");
    let borrowed_cow: Cow<'_, Path> = borrowed.into();
    let owned_cow: Cow<'_, Path> = owned.clone().into();

    macro_rules! t {
        ($($current:expr),+) => {
            $(
                assert_eq!($current, borrowed);
                assert_eq!($current, owned);
                assert_eq!($current, borrowed_cow);
                assert_eq!($current, owned_cow);
            )+
        }
    }

    t!(borrowed, owned, borrowed_cow, owned_cow);
}

#[test]
pub fn test_compare() {
    use crate::collections::hash_map::DefaultHasher;
    use crate::hash::{Hash, Hasher};

    fn hash<T: Hash>(t: T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    macro_rules! tc (
        ($path1:expr, $path2:expr, eq: $eq:expr,
         starts_with: $starts_with:expr, ends_with: $ends_with:expr,
         relative_from: $relative_from:expr) => ({
             let path1 = Path::new($path1);
             let path2 = Path::new($path2);

             let eq = path1 == path2;
             assert!(eq == $eq, "{:?} == {:?}, expected {:?}, got {:?}",
                     $path1, $path2, $eq, eq);
             assert!($eq == (hash(path1) == hash(path2)),
                     "{:?} == {:?}, expected {:?}, got {} and {}",
                     $path1, $path2, $eq, hash(path1), hash(path2));

             let starts_with = path1.starts_with(path2);
             assert!(starts_with == $starts_with,
                     "{:?}.starts_with({:?}), expected {:?}, got {:?}", $path1, $path2,
                     $starts_with, starts_with);

             let ends_with = path1.ends_with(path2);
             assert!(ends_with == $ends_with,
                     "{:?}.ends_with({:?}), expected {:?}, got {:?}", $path1, $path2,
                     $ends_with, ends_with);

             let relative_from = path1.strip_prefix(path2)
                                      .map(|p| p.to_str().unwrap())
                                      .ok();
             let exp: Option<&str> = $relative_from;
             assert!(relative_from == exp,
                     "{:?}.strip_prefix({:?}), expected {:?}, got {:?}",
                     $path1, $path2, exp, relative_from);
        });
    );

    tc!("", "",
    eq: true,
    starts_with: true,
    ends_with: true,
    relative_from: Some("")
    );

    tc!("foo", "",
    eq: false,
    starts_with: true,
    ends_with: true,
    relative_from: Some("foo")
    );

    tc!("", "foo",
    eq: false,
    starts_with: false,
    ends_with: false,
    relative_from: None
    );

    tc!("foo", "foo",
    eq: true,
    starts_with: true,
    ends_with: true,
    relative_from: Some("")
    );

    tc!("foo/", "foo",
    eq: true,
    starts_with: true,
    ends_with: true,
    relative_from: Some("")
    );

    tc!("foo/.", "foo",
    eq: true,
    starts_with: true,
    ends_with: true,
    relative_from: Some("")
    );

    tc!("foo/./bar", "foo/bar",
    eq: true,
    starts_with: true,
    ends_with: true,
    relative_from: Some("")
    );

    tc!("foo/bar", "foo",
    eq: false,
    starts_with: true,
    ends_with: false,
    relative_from: Some("bar")
    );

    tc!("foo/bar/baz", "foo/bar",
    eq: false,
    starts_with: true,
    ends_with: false,
    relative_from: Some("baz")
    );

    tc!("foo/bar", "foo/bar/baz",
    eq: false,
    starts_with: false,
    ends_with: false,
    relative_from: None
    );

    tc!("./foo/bar/", ".",
    eq: false,
    starts_with: true,
    ends_with: false,
    relative_from: Some("foo/bar")
    );

    if cfg!(windows) {
        tc!(r"C:\src\rust\cargo-test\test\Cargo.toml",
        r"c:\src\rust\cargo-test\test",
        eq: false,
        starts_with: true,
        ends_with: false,
        relative_from: Some("Cargo.toml")
        );

        tc!(r"c:\foo", r"C:\foo",
        eq: true,
        starts_with: true,
        ends_with: true,
        relative_from: Some("")
        );

        tc!(r"C:\foo\.\bar.txt", r"C:\foo\bar.txt",
        eq: true,
        starts_with: true,
        ends_with: true,
        relative_from: Some("")
        );

        tc!(r"C:\foo\.", r"C:\foo",
        eq: true,
        starts_with: true,
        ends_with: true,
        relative_from: Some("")
        );

        tc!(r"\\?\C:\foo\.\bar.txt", r"\\?\C:\foo\bar.txt",
        eq: false,
        starts_with: false,
        ends_with: false,
        relative_from: None
        );
    }
}

#[test]
fn test_components_debug() {
    let path = Path::new("/tmp");

    let mut components = path.components();

    let expected = "Components([RootDir, Normal(\"tmp\")])";
    let actual = format!("{:?}", components);
    assert_eq!(expected, actual);

    let _ = components.next().unwrap();
    let expected = "Components([Normal(\"tmp\")])";
    let actual = format!("{:?}", components);
    assert_eq!(expected, actual);

    let _ = components.next().unwrap();
    let expected = "Components([])";
    let actual = format!("{:?}", components);
    assert_eq!(expected, actual);
}

#[cfg(unix)]
#[test]
fn test_iter_debug() {
    let path = Path::new("/tmp");

    let mut iter = path.iter();

    let expected = "Iter([\"/\", \"tmp\"])";
    let actual = format!("{:?}", iter);
    assert_eq!(expected, actual);

    let _ = iter.next().unwrap();
    let expected = "Iter([\"tmp\"])";
    let actual = format!("{:?}", iter);
    assert_eq!(expected, actual);

    let _ = iter.next().unwrap();
    let expected = "Iter([])";
    let actual = format!("{:?}", iter);
    assert_eq!(expected, actual);
}

#[test]
fn into_boxed() {
    let orig: &str = "some/sort/of/path";
    let path = Path::new(orig);
    let boxed: Box<Path> = Box::from(path);
    let path_buf = path.to_owned().into_boxed_path().into_path_buf();
    assert_eq!(path, &*boxed);
    assert_eq!(&*boxed, &*path_buf);
    assert_eq!(&*path_buf, path);
}

#[test]
fn test_clone_into() {
    let mut path_buf = PathBuf::from("supercalifragilisticexpialidocious");
    let path = Path::new("short");
    path.clone_into(&mut path_buf);
    assert_eq!(path, path_buf);
    assert!(path_buf.into_os_string().capacity() >= 15);
}

#[test]
fn display_format_flags() {
    assert_eq!(format!("a{:#<5}b", Path::new("").display()), "a#####b");
    assert_eq!(format!("a{:#<5}b", Path::new("a").display()), "aa####b");
}

#[test]
fn into_rc() {
    let orig = "hello/world";
    let path = Path::new(orig);
    let rc: Rc<Path> = Rc::from(path);
    let arc: Arc<Path> = Arc::from(path);

    assert_eq!(&*rc, path);
    assert_eq!(&*arc, path);

    let rc2: Rc<Path> = Rc::from(path.to_owned());
    let arc2: Arc<Path> = Arc::from(path.to_owned());

    assert_eq!(&*rc2, path);
    assert_eq!(&*arc2, path);
}

#[test]
fn test_ord() {
    macro_rules! ord(
        ($ord:ident, $left:expr, $right:expr) => ( {
            use core::cmp::Ordering;

            let left = Path::new($left);
            let right = Path::new($right);
            assert_eq!(left.cmp(&right), Ordering::$ord);
            if (core::cmp::Ordering::$ord == Ordering::Equal) {
                assert_eq!(left, right);

                let mut hasher = DefaultHasher::new();
                left.hash(&mut hasher);
                let left_hash = hasher.finish();
                hasher = DefaultHasher::new();
                right.hash(&mut hasher);
                let right_hash = hasher.finish();

                assert_eq!(left_hash, right_hash, "hashes for {:?} and {:?} must match", left, right);
            } else {
                assert_ne!(left, right);
            }
        });
    );

    ord!(Less, "1", "2");
    ord!(Less, "/foo/bar", "/foo./bar");
    ord!(Less, "foo/bar", "foo/bar.");
    ord!(Equal, "foo/./bar", "foo/bar/");
    ord!(Equal, "foo/bar", "foo/bar/");
    ord!(Equal, "foo/bar", "foo/bar/.");
    ord!(Equal, "foo/bar", "foo/bar//");
}

#[test]
#[cfg(unix)]
fn test_unix_absolute() {
    use crate::path::absolute;

    assert!(absolute("").is_err());

    let relative = "a/b";
    let mut expected = crate::env::current_dir().unwrap();
    expected.push(relative);
    assert_eq!(absolute(relative).unwrap(), expected);

    // Test how components are collected.
    assert_eq!(absolute("/a/b/c").unwrap(), Path::new("/a/b/c"));
    assert_eq!(absolute("/a//b/c").unwrap(), Path::new("/a/b/c"));
    assert_eq!(absolute("//a/b/c").unwrap(), Path::new("//a/b/c"));
    assert_eq!(absolute("///a/b/c").unwrap(), Path::new("/a/b/c"));
    assert_eq!(absolute("/a/b/c/").unwrap(), Path::new("/a/b/c/"));
    assert_eq!(absolute("/a/./b/../c/.././..").unwrap(), Path::new("/a/b/../c/../.."));
}

#[test]
#[cfg(windows)]
fn test_windows_absolute() {
    use crate::path::absolute;
    // An empty path is an error.
    assert!(absolute("").is_err());

    let relative = r"a\b";
    let mut expected = crate::env::current_dir().unwrap();
    expected.push(relative);
    assert_eq!(absolute(relative).unwrap(), expected);

    macro_rules! unchanged(
        ($path:expr) => {
            assert_eq!(absolute($path).unwrap(), Path::new($path));
        }
    );

    unchanged!(r"C:\path\to\file");
    unchanged!(r"C:\path\to\file\");
    unchanged!(r"\\server\share\to\file");
    unchanged!(r"\\server.\share.\to\file");
    unchanged!(r"\\.\PIPE\name");
    unchanged!(r"\\.\C:\path\to\COM1");
    unchanged!(r"\\?\C:\path\to\file");
    unchanged!(r"\\?\UNC\server\share\to\file");
    unchanged!(r"\\?\PIPE\name");
    // Verbatim paths are always unchanged, no matter what.
    unchanged!(r"\\?\path.\to/file..");

    assert_eq!(absolute(r"C:\path..\to.\file.").unwrap(), Path::new(r"C:\path..\to\file"));
    assert_eq!(absolute(r"C:\path\to\COM1").unwrap(), Path::new(r"\\.\COM1"));
    assert_eq!(absolute(r"C:\path\to\COM1.txt").unwrap(), Path::new(r"\\.\COM1"));
    assert_eq!(absolute(r"C:\path\to\COM1  .txt").unwrap(), Path::new(r"\\.\COM1"));
    assert_eq!(absolute(r"C:\path\to\cOnOuT$").unwrap(), Path::new(r"\\.\cOnOuT$"));
}

#[bench]
fn bench_path_cmp_fast_path_buf_sort(b: &mut test::Bencher) {
    let prefix = "my/home";
    let mut paths: Vec<_> =
        (0..1000).map(|num| PathBuf::from(prefix).join(format!("file {}.rs", num))).collect();

    paths.sort();

    b.iter(|| {
        black_box(paths.as_mut_slice()).sort_unstable();
    });
}

#[bench]
fn bench_path_cmp_fast_path_long(b: &mut test::Bencher) {
    let prefix = "/my/home/is/my/castle/and/my/castle/has/a/rusty/workbench/";
    let paths: Vec<_> =
        (0..1000).map(|num| PathBuf::from(prefix).join(format!("file {}.rs", num))).collect();

    let mut set = BTreeSet::new();

    paths.iter().for_each(|p| {
        set.insert(p.as_path());
    });

    b.iter(|| {
        set.remove(paths[500].as_path());
        set.insert(paths[500].as_path());
    });
}

#[bench]
fn bench_path_cmp_fast_path_short(b: &mut test::Bencher) {
    let prefix = "my/home";
    let paths: Vec<_> =
        (0..1000).map(|num| PathBuf::from(prefix).join(format!("file {}.rs", num))).collect();

    let mut set = BTreeSet::new();

    paths.iter().for_each(|p| {
        set.insert(p.as_path());
    });

    b.iter(|| {
        set.remove(paths[500].as_path());
        set.insert(paths[500].as_path());
    });
}

#[bench]
fn bench_path_hashset(b: &mut test::Bencher) {
    let prefix = "/my/home/is/my/castle/and/my/castle/has/a/rusty/workbench/";
    let paths: Vec<_> =
        (0..1000).map(|num| PathBuf::from(prefix).join(format!("file {}.rs", num))).collect();

    let mut set = HashSet::new();

    paths.iter().for_each(|p| {
        set.insert(p.as_path());
    });

    b.iter(|| {
        set.remove(paths[500].as_path());
        set.insert(black_box(paths[500].as_path()))
    });
}

#[bench]
fn bench_path_hashset_miss(b: &mut test::Bencher) {
    let prefix = "/my/home/is/my/castle/and/my/castle/has/a/rusty/workbench/";
    let paths: Vec<_> =
        (0..1000).map(|num| PathBuf::from(prefix).join(format!("file {}.rs", num))).collect();

    let mut set = HashSet::new();

    paths.iter().for_each(|p| {
        set.insert(p.as_path());
    });

    let probe = PathBuf::from(prefix).join("other");

    b.iter(|| set.remove(black_box(probe.as_path())));
}

#[bench]
fn bench_hash_path_short(b: &mut test::Bencher) {
    let mut hasher = DefaultHasher::new();
    let path = Path::new("explorer.exe");

    b.iter(|| black_box(path).hash(&mut hasher));

    black_box(hasher.finish());
}

#[bench]
fn bench_hash_path_long(b: &mut test::Bencher) {
    let mut hasher = DefaultHasher::new();
    let path =
        Path::new("/aaaaa/aaaaaa/./../aaaaaaaa/bbbbbbbbbbbbb/ccccccccccc/ddddddddd/eeeeeee.fff");

    b.iter(|| black_box(path).hash(&mut hasher));

    black_box(hasher.finish());
}
