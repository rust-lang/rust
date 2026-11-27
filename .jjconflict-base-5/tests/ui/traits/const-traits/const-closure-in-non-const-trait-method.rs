#![feature(const_closures)]

// Regression test for https://github.com/rust-lang/rust/issues/153891

trait Tr {
    const fn test() {
        //~^ ERROR functions in traits cannot be declared const
        (const || {})()
        //~^ ERROR cannot use `const` closures outside of const contexts
    }
}

fn main() {}
