#![feature(rustc_attrs)]

// This test is the same code as in ui/issue-53912.rs but this test checks that the symbol mangling
// fix produces the correct result, whereas that test just checks that the reproduction compiles
// successfully and doesn't segfault

fn dummy() {}

mod llvm {
    pub(crate) struct Foo;
}
mod foo {
    pub(crate) struct Foo<T>(T);

    impl Foo<::llvm::Foo> {
        #[rustc_symbol_name]
//~^ ERROR _ZN11issue_609253foo36Foo$LT$issue_60925..llv$6d$..Foo$GT$3foo17h059a991a004536adE
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
