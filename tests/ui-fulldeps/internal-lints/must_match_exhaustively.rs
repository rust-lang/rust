//@ compile-flags: -Z unstable-options
//@ ignore-stage1

#![feature(rustc_private)]
#![feature(rustc_attrs)]
#![deny(rustc::rustc_must_match_exhaustively)]


#[rustc_must_match_exhaustively]
#[derive(Copy, Clone)]
enum Foo {
    A {field: u32},
    B,
}

fn foo(f: Foo) {
    match f {
        Foo::A {..}=> {}
        Foo::B => {}
    }

    match f {
        //~^ ERROR match is not exhaustive
        Foo::A {..} => {}
        _ => {}
    }

    match f {
        //~^ ERROR match is not exhaustive
        Foo::A {..} => {}
        a => {}
    }

    match &f {
        //~^ ERROR match is not exhaustive
        Foo::A {..} => {}
        a => {}
    }

    match f {
        Foo::A {..} => {}
        a@Foo::B => {}
    }

    if let Foo::A {..} = f {}
    //~^ ERROR match is not exhaustive
}

fn main() {}
