// This test case tests whether we can handle code bases that contain a high
// number of closures, something that needs special handling in the MingGW
// toolchain.
// See https://github.com/rust-lang/rust/issues/34793 for more information.

// Make sure we don't optimize anything away:
// compile-flags: -C no-prepopulate-passes -Cpasses=name-anon-globals

// Expand something exponentially
macro_rules! go_bacterial {
    ($mac:ident) => ($mac!());
    ($mac:ident 1 $($t:tt)*) => (
        go_bacterial!($mac $($t)*);
        go_bacterial!($mac $($t)*);
    )
}

macro_rules! mk_closure {
    () => ((move || {})())
}

macro_rules! mk_fn {
    () => {
        {
            fn function() {
                // Make 16 closures
                go_bacterial!(mk_closure 1 1 1 1);
            }
            let _ = function();
        }
    }
}

fn main() {
    // Make 2^8 functions, each containing 16 closures,
    // resulting in 2^12 closures overall.
    go_bacterial!(mk_fn 1 1 1 1  1 1 1 1);
}
