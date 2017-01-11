#![feature(rustc_attrs)]

struct Foo {
    x: u8
}

impl Foo {
    // Changing the item `new`...
    #[rustc_if_this_changed(HirBody)]
    fn new() -> Foo {
        Foo { x: 0 }
    }

    // ...should not cause us to recompute the tables for `with`!
    #[rustc_then_this_would_need(Tables)] //~ ERROR no path
    fn with(x: u8) -> Foo {
        Foo { x: x }
    }
}

fn main() {
    let f = Foo::new();
    let g = Foo::with(22);
    assert_eq!(f.x, g.x - 22);
}
