// ignore-tidy-linelength
// revisions: legacy v0
//[legacy]compile-flags: -Z symbol-mangling-version=legacy
    //[v0]compile-flags: -Z symbol-mangling-version=v0

#![feature(rustc_attrs)]

// This test is the same code as in ui/issue-53912.rs but this test checks that the symbol mangling
// fix produces the correct result, whereas that test just checks that the reproduction compiles
// successfully and doesn't crash LLVM

fn dummy() {}

mod llvm {
    pub(crate) struct Foo;
}
mod foo {
    pub(crate) struct Foo<T>(T);

    impl Foo<::llvm::Foo> {
        #[rustc_symbol_name]
        //[legacy]~^ ERROR symbol-name(_ZN11issue_609253foo37Foo$LT$issue_60925..llv$u6d$..Foo$GT$3foo
        //[legacy]~| ERROR demangling(issue_60925::foo::Foo<issue_60925::llv$u6d$..Foo$GT$::foo
        //[legacy]~| ERROR demangling-alt(issue_60925::foo::Foo<issue_60925::llv$u6d$..Foo$GT$::foo)
         //[v0]~^^^^ ERROR symbol-name(_RNvMNtCs4fqI2P2rA04_11issue_609253fooINtB2_3FooNtNtB4_4llvm3FooE3foo)
            //[v0]~| ERROR demangling(<issue_60925[317d481089b8c8fe]::foo::Foo<issue_60925[317d481089b8c8fe]::llvm::Foo>>::foo)
            //[v0]~| ERROR demangling-alt(<issue_60925::foo::Foo<issue_60925::llvm::Foo>>::foo)
        pub(crate) fn foo() {
            for _ in 0..0 {
                for _ in &[::dummy()] {
                    ::dummy();
                    ::dummy();
                    ::dummy();
                }
            }
        }
    }

    pub(crate) fn foo() {
        Foo::foo();
        Foo::foo();
    }
}

pub fn foo() {
    foo::foo();
}

fn main() {}
