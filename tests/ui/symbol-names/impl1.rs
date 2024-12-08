//@ build-fail
//@ revisions: legacy v0
//@[legacy]compile-flags: -Z unstable-options -C symbol-mangling-version=legacy
//@[v0]compile-flags: -C symbol-mangling-version=v0
//@[legacy]normalize-stderr-test: "h[\w]{16}E?\)" -> "<SYMBOL_HASH>)"

#![feature(auto_traits, rustc_attrs)]
#![allow(dead_code)]

mod foo {
    pub struct Foo { x: u32 }

    impl Foo {
        #[rustc_symbol_name]
        //[legacy]~^ ERROR symbol-name(_ZN5impl13foo3Foo3bar
        //[legacy]~| ERROR demangling(impl1::foo::Foo::bar
        //[legacy]~| ERROR demangling-alt(impl1::foo::Foo::bar)
         //[v0]~^^^^ ERROR symbol-name(_RNvMNtCs
            //[v0]~| ERROR demangling(<impl1[
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
         //[v0]~^^^^ ERROR symbol-name(_RNvMNtCs
            //[v0]~| ERROR demangling(<impl1[
            //[v0]~| ERROR demangling-alt(<impl1::foo::Foo>::baz)
        #[rustc_def_path]
        //[legacy]~^ ERROR def-path(bar::<impl foo::Foo>::baz)
           //[v0]~^^ ERROR def-path(bar::<impl foo::Foo>::baz)
        fn baz() { }
    }
}

trait Foo {
    type Assoc;
}

auto trait AutoTrait {}

fn main() {
    // Test closure mangling, and disambiguators.
    || {};
    || {
        trait Bar {
            fn method(&self) {}
        }

        // Test type mangling, by putting them in an `impl` header.
        impl Bar for [&'_ (dyn Foo<Assoc = extern "C" fn(&u8, ...)> + AutoTrait); 3] {
            #[rustc_symbol_name]
            //[legacy]~^ ERROR symbol-name(_ZN209_$LT$$u5b$$RF$dyn$u20$impl1..Foo$u2b$Assoc$u20$$u3d$$u20$extern$u20$$u22$C$u22$$u20$fn$LP$$RF$u8$C$$u20$...$RP$$u2b$impl1..AutoTrait$u3b$$u20$3$u5d$$u20$as$u20$impl1..main..$u7b$$u7b$closure$u7d$$u7d$..Bar$GT$6method
            //[legacy]~| ERROR demangling(<[&dyn impl1::Foo+Assoc = extern "C" fn(&u8, ::.)+impl1::AutoTrait; 3] as impl1::main::{{closure}}::Bar>::method
            //[legacy]~| ERROR demangling-alt(<[&dyn impl1::Foo+Assoc = extern "C" fn(&u8, ::.)+impl1::AutoTrait; 3] as impl1::main::{{closure}}::Bar>::method)
             //[v0]~^^^^ ERROR symbol-name(_RNvXNCNvCs
                //[v0]~| ERROR demangling(<[&dyn
                //[v0]~| ERROR demangling-alt(<[&dyn impl1::Foo<Assoc = for<'a> extern "C" fn(&'a u8, ...)> + impl1::AutoTrait; 3] as impl1::main::{closure#1}::Bar>::method)
            #[rustc_def_path]
            //[legacy]~^ ERROR def-path(<[&dyn Foo<Assoc = for<'a> extern "C" fn(&'a u8, ...)> + AutoTrait; 3] as main::{closure#1}::Bar>::method)
               //[v0]~^^ ERROR def-path(<[&dyn Foo<Assoc = for<'a> extern "C" fn(&'a u8, ...)> + AutoTrait; 3] as main::{closure#1}::Bar>::method)
            fn method(&self) {}
        }
    };
}
