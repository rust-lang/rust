//@ build-fail
//@ revisions: legacy v0
//@[legacy]compile-flags: -Z unstable-options -C symbol-mangling-version=legacy
//@[v0]compile-flags: -C symbol-mangling-version=v0

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

    impl Foo<crate::llvm::Foo> {
        #[rustc_symbol_name]
        //[legacy]~^ ERROR symbol-name(_ZN11issue_609253foo37Foo$LT$issue_60925..llv$u6d$..Foo$GT$3foo
        //[legacy]~| ERROR demangling(issue_60925::foo::Foo<issue_60925::llvm::Foo>::foo
        //[legacy]~| ERROR demangling-alt(issue_60925::foo::Foo<issue_60925::llvm::Foo>::foo)
         //[v0]~^^^^ ERROR symbol-name
            //[v0]~| ERROR demangling
            //[v0]~| ERROR demangling-alt(<issue_60925::foo::Foo<issue_60925::llvm::Foo>>::foo)
        pub(crate) fn foo() {
            for _ in 0..0 {
                for _ in &[crate::dummy()] {
                    crate::dummy();
                    crate::dummy();
                    crate::dummy();
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
