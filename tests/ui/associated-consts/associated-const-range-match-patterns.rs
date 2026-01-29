//@ revisions: edition2015 edition2018
//@[edition2015]edition:2015
//@[edition2018]edition:2018..
//@run-pass
#![allow(dead_code, unreachable_patterns)]
#![allow(ellipsis_inclusive_range_patterns)]

struct Foo;

trait HasNum {
    const NUM: isize;
}
impl HasNum for Foo {
    const NUM: isize = 1;
}

fn main() {
    #[cfg(edition2015)]
    assert!(match 2 {
        Foo::NUM ... 3 => true,
        _ => false,
    });
    #[cfg(edition2015)]
    assert!(match 0 {
        -1 ... <Foo as HasNum>::NUM => true,
        _ => false,
    });
    #[cfg(edition2015)]
    assert!(match 1 {
        <Foo as HasNum>::NUM ... <Foo>::NUM => true,
        _ => false,
    });

    assert!(match 2 {
        Foo::NUM ..= 3 => true,
        _ => false,
    });
    assert!(match 0 {
        -1 ..= <Foo as HasNum>::NUM => true,
        _ => false,
    });
    assert!(match 1 {
        <Foo as HasNum>::NUM ..= <Foo>::NUM => true,
        _ => false,
    });
}
