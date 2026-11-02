//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass
//@ edition: 2018
//@ run-rustfix

#![feature(try_blocks)]
#![warn(unused_parens, unused_braces)]

fn consume<T>(_: Result<T, T>) -> T { todo!() }

fn main() {
    consume((try {}));
    //~^ WARN unnecessary parentheses

    consume({ try {} });
    //~^ WARN unnecessary braces

    match (try {}) {
        //~^ WARN unnecessary parentheses
        Ok(()) | Err(()) => (),
    }

    if let Err(()) = (try {}) {}
    //~^ WARN unnecessary parentheses

    match (try {}) {
        //~^ WARN unnecessary parentheses
        Ok(()) | Err(()) => (),
    }
}
