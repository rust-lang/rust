// build-pass (FIXME(62277): could be check-pass?)

// This test is the same code as in ui/symbol-names/issue-60925.rs but this checks that the
// reproduction compiles successfully and doesn't segfault, whereas that test just checks that the
// symbol mangling fix produces the correct result.

fn dummy() {}

mod llvm {
    pub(crate) struct Foo;
}
mod foo {
    pub(crate) struct Foo<T>(T);

    impl Foo<::llvm::Foo> {
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
