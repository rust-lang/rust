#![feature(rustc_attrs)]
#![allow(dead_code)]

mod foo {
    pub struct Foo { x: u32 }

    impl Foo {
        #[rustc_symbol_name]
        //~^ ERROR symbol-name(_ZN5impl13foo3Foo3bar
        //~| ERROR demangling(impl1::foo::Foo::bar
        //~| ERROR demangling-alt(impl1::foo::Foo::bar)
        #[rustc_def_path] //~ ERROR def-path(foo::Foo::bar)
        fn bar() { }
    }
}

mod bar {
    use foo::Foo;

    impl Foo {
        #[rustc_symbol_name]
        //~^ ERROR symbol-name(_ZN5impl13bar33_$LT$impl$u20$impl1..foo..Foo$GT$3baz
        //~| ERROR demangling(impl1::bar::<impl impl1::foo::Foo>::baz
        //~| ERROR demangling-alt(impl1::bar::<impl impl1::foo::Foo>::baz)
        #[rustc_def_path] //~ ERROR def-path(bar::<impl foo::Foo>::baz)
        fn baz() { }
    }
}

fn main() {
}
