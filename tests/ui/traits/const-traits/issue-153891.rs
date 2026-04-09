// Regression test for issue #153891.
#![feature(const_closures)]

trait Tr {
    const fn test() {
        //~^ ERROR functions in traits cannot be declared const
        (const || {})()
    }
}

fn main() {}
