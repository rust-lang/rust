//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
use std::convert::identity;

fn test<'a: 'a>(n: bool) -> impl Sized + 'a {
    let true = n else { loop {} };
    let _ = || {
        let _ = identity::<&'a ()>(test(false));
        //[current]~^ ERROR concrete type differs from previous defining opaque type use
    };
    loop {}
}

fn main() {}
