// Test for issue #153891: ICE when using const closure in trait const fn
#![feature(const_closures)]

trait Tr {
    const fn test() {
        (const || {})()
    }
}

fn main() {}
