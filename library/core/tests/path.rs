#[cfg(target_os = "windows")]
mod windows;

use crate::collections::{BTreeSet, HashSet};
use crate::hash::DefaultHasher;
use crate::hint::black_box;
use crate::mem::MaybeUninit;
use crate::path::*;
use crate::ptr;

#[test]
pub fn test_compare() {
    use crate::hash::{DefaultHasher, Hash, Hasher};

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

    tc!("foo//", "foo",
    eq: true,
    starts_with: true,
    ends_with: true,
    relative_from: Some("")
    );

    tc!("foo///", "foo",
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

    tc!("foo/.//bar", "foo/bar",
    eq: true,
    starts_with: true,
    ends_with: true,
    relative_from: Some("")
    );

    tc!("foo//./bar", "foo/bar",
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

    tc!("foo/bar", "foobar",
    eq: false,
    starts_with: false,
    ends_with: false,
    relative_from: None
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
    let actual = format!("{components:?}");
    assert_eq!(expected, actual);

    let _ = components.next().unwrap();
    let expected = "Components([Normal(\"tmp\")])";
    let actual = format!("{components:?}");
    assert_eq!(expected, actual);

    let _ = components.next().unwrap();
    let expected = "Components([])";
    let actual = format!("{components:?}");
    assert_eq!(expected, actual);
}

#[cfg(unix)]
#[test]
fn test_iter_debug() {
    let path = Path::new("/tmp");

    let mut iter = path.iter();

    let expected = "Iter([\"/\", \"tmp\"])";
    let actual = format!("{iter:?}");
    assert_eq!(expected, actual);

    let _ = iter.next().unwrap();
    let expected = "Iter([\"tmp\"])";
    let actual = format!("{iter:?}");
    assert_eq!(expected, actual);

    let _ = iter.next().unwrap();
    let expected = "Iter([])";
    let actual = format!("{iter:?}");
    assert_eq!(expected, actual);
}

#[test]
fn display_format_flags() {
    assert_eq!(format!("a{:#<5}b", Path::new("").display()), "a#####b");
    assert_eq!(format!("a{:#<5}b", Path::new("a").display()), "aa####b");
}

#[test]
fn test_ord() {
    macro_rules! ord(
        ($ord:ident, $left:expr, $right:expr) => ({
            use crate::cmp::Ordering;

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

#[test]
fn clone_to_uninit() {
    let a = Path::new("hello.txt");

    let mut storage = vec![MaybeUninit::<u8>::uninit(); size_of_val::<Path>(a)];
    unsafe { a.clone_to_uninit(ptr::from_mut::<[_]>(storage.as_mut_slice()) as *mut Path) };
    assert_eq!(a.as_os_str().as_encoded_bytes(), unsafe {
        MaybeUninit::slice_assume_init_ref(&storage)
    });

    let mut b: Box<Path> = Path::new("world.exe").into();
    assert_eq!(size_of_val::<Path>(a), size_of_val::<Path>(&b));
    assert_ne!(a, &*b);
    unsafe { a.clone_to_uninit(ptr::from_mut::<Path>(&mut b)) };
    assert_eq!(a, &*b);
}
