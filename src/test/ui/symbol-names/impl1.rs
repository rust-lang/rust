// revisions: legacy v0
//[legacy]compile-flags: -Z symbol-mangling-version=legacy
    //[v0]compile-flags: -Z symbol-mangling-version=v0

#![feature(rustc_attrs)]
#![allow(dead_code)]

mod foo {
    pub struct Foo { x: u32 }

    impl Foo {
        #[rustc_symbol_name]
        //[legacy]~^ ERROR symbol-name(_ZN5impl13foo3Foo3bar
        //[legacy]~| ERROR demangling(impl1::foo::Foo::bar
        //[legacy]~| ERROR demangling-alt(impl1::foo::Foo::bar)
         //[v0]~^^^^ ERROR symbol-name(_RNvMNtCs4fqI2P2rA04_5impl13fooNtB2_3Foo3bar)
            //[v0]~| ERROR demangling(<impl1[317d481089b8c8fe]::foo::Foo>::bar)
            //[v0]~| ERROR demangling-alt(<impl1::foo::Foo>::bar)
        #[rustc_def_path]
        //[legacy]~^ ERROR def-path(foo::Foo::bar)
           //[v0]~^^ ERROR def-path(foo::Foo::bar)
        fn bar() { }
    }
}

mod bar {
    use foo::Foo;

    impl Foo {
        #[rustc_symbol_name]
        //[legacy]~^ ERROR symbol-name(_ZN5impl13bar33_$LT$impl$u20$impl1..foo..Foo$GT$3baz
        //[legacy]~| ERROR demangling(impl1::bar::<impl impl1::foo::Foo>::baz
        //[legacy]~| ERROR demangling-alt(impl1::bar::<impl impl1::foo::Foo>::baz)
         //[v0]~^^^^ ERROR symbol-name(_RNvMNtCs4fqI2P2rA04_5impl13barNtNtB4_3foo3Foo3baz)
            //[v0]~| ERROR demangling(<impl1[317d481089b8c8fe]::foo::Foo>::baz)
            //[v0]~| ERROR demangling-alt(<impl1::foo::Foo>::baz)
        #[rustc_def_path]
        //[legacy]~^ ERROR def-path(bar::<impl foo::Foo>::baz)
           //[v0]~^^ ERROR def-path(bar::<impl foo::Foo>::baz)
        fn baz() { }
    }
}

fn main() {
}
