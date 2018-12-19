#![feature(rustc_attrs)]
#![allow(dead_code)]

mod foo {
    pub struct Foo { x: u32 }

    impl Foo {
        #[rustc_symbol_name] //~ ERROR _ZN15impl1..foo..Foo3bar
        #[rustc_def_path] //~ ERROR def-path(foo::Foo::bar)
        fn bar() { }
    }
}

mod bar {
    use foo::Foo;

    impl Foo {
        #[rustc_symbol_name] //~ ERROR _ZN5impl13bar33_$LT$impl$u20$impl1..foo..Foo$GT$3baz
        #[rustc_def_path] //~ ERROR def-path(bar::<impl foo::Foo>::baz)
        fn baz() { }
    }
}

fn main() {
}
