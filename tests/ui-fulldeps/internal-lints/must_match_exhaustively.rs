//@ compile-flags: -Z unstable-options
//@ ignore-stage1

#![feature(rustc_private)]
#![feature(register_tool)]
#![feature(rustc_attrs)]
#![register_tool(lint)]

#![deny(rustc::must_match_exhaustively)]

#[lint::must_match_exhaustively]
#[derive(Copy, Clone)]
enum Foo {
    A { field: u32 },
    B,
}

fn foo(f: Foo) {
    match f {
        Foo::A { .. } => {}
        Foo::B => {}
    }

    match f {
        //~^ ERROR match is not exhaustive
        Foo::A { .. } => {}
        _ => {}
    }

    match f {
        //~^ ERROR match is not exhaustive
        Foo::A { .. } => {}
        a => {}
    }

    match &f {
        //~^ ERROR match is not exhaustive
        Foo::A { .. } => {}
        a => {}
    }

    match f {
        Foo::A { .. } => {}
        a @ Foo::B => {}
    }

    if let Foo::A { .. } = f {}
    //~^ ERROR match is not exhaustive

    let Foo::A { .. } = f else { loop {} };
    //~^ ERROR match is not exhaustive

    matches!(f, Foo::A { .. });
    //~^ ERROR match is not exhaustive
}

fn main() {}
