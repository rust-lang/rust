use core::hint::black_box;
use std::collections::{BTreeSet, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::*;

#[bench]
#[cfg_attr(miri, ignore)] // Miri isn't fast...
fn bench_path_cmp_fast_path_buf_sort(b: &mut test::Bencher) {
    let prefix = "my/home";
    let mut paths: Vec<_> =
        (0..1000).map(|num| PathBuf::from(prefix).join(format!("file {num}.rs"))).collect();

    paths.sort();

    b.iter(|| {
        black_box(paths.as_mut_slice()).sort_unstable();
    });
}

#[bench]
#[cfg_attr(miri, ignore)] // Miri isn't fast...
fn bench_path_cmp_fast_path_long(b: &mut test::Bencher) {
    let prefix = "/my/home/is/my/castle/and/my/castle/has/a/rusty/workbench/";
    let paths: Vec<_> =
        (0..1000).map(|num| PathBuf::from(prefix).join(format!("file {num}.rs"))).collect();

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
#[cfg_attr(miri, ignore)] // Miri isn't fast...
fn bench_path_cmp_fast_path_short(b: &mut test::Bencher) {
    let prefix = "my/home";
    let paths: Vec<_> =
        (0..1000).map(|num| PathBuf::from(prefix).join(format!("file {num}.rs"))).collect();

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
#[cfg_attr(miri, ignore)] // Miri isn't fast...
fn bench_path_hashset(b: &mut test::Bencher) {
    let prefix = "/my/home/is/my/castle/and/my/castle/has/a/rusty/workbench/";
    let paths: Vec<_> =
        (0..1000).map(|num| PathBuf::from(prefix).join(format!("file {num}.rs"))).collect();

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
#[cfg_attr(miri, ignore)] // Miri isn't fast...
fn bench_path_hashset_miss(b: &mut test::Bencher) {
    let prefix = "/my/home/is/my/castle/and/my/castle/has/a/rusty/workbench/";
    let paths: Vec<_> =
        (0..1000).map(|num| PathBuf::from(prefix).join(format!("file {num}.rs"))).collect();

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
