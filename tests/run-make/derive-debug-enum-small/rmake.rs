use run_make_support::{bare_rustc, rfs};

fn main() {
    bare_rustc()
        .input("test.rs")
        .arg("-Copt-level=s")
        .arg("-Cpanic=abort")
        .arg("-Clink-arg=-nostartfiles")
        .arg("-Clink-arg=-lc")
        .arg("-Cdebuginfo=0")
        .arg("-Cstrip=symbols")
        .output("test")
        .run();

    let binary_size = rfs::read("test").len();

    // At the time of writing, the binary_size is 8 000 on Linux.
    // The original issue was bloat in `derive(Debug)` for
    // c-like enums, caused by combination of:
    //  - string indexing, which makes it jump to 18 312
    //  -` .split(' ')` iterator, which makes it jump to 9 160
    let threshold = 8_400;
    assert!(
        binary_size < threshold,
        "binary_size was: {binary_size}, expected less than {threshold}"
    );
}
