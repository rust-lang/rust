#![allow(dead_code, unreachable_code)]
//@ edition: 2021

// Regression test for #93054: Functions using uninhabited types often only have a single,
// unreachable basic block which doesn't get instrumented. This should not cause llvm-cov to fail.
// Since these kinds functions can't be invoked anyway, it's ok to not have coverage data for them.

enum Never {}

impl Never {
    fn foo(self) {
        match self {}
        make().map(|never| match never {});
    }

    fn bar(&self) {
        match *self {}
    }
}

async fn foo2(never: Never) {
    match never {}
}

fn make() -> Option<Never> {
    None
}

fn main() {}
