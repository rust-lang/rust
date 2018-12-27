// run-pass
#![allow(dead_code)]
// This test case exposes conditions where the encoding of a trait object type
// with projection predicates would differ between this crate and the upstream
// crate, because the predicates were encoded in different order within each
// crate. This led to different symbol hashes of functions using these type,
// which in turn led to linker errors because the two crates would not agree on
// the symbol name.
// The fix was to make the order in which predicates get encoded stable.

// aux-build:issue34796aux.rs
extern crate issue34796aux;

fn mk<T>() -> T { loop {} }

struct Data<T, E> {
    data: T,
    error: E,
}

fn main() {
    issue34796aux::bar(|()| {
        Data::<(), std::io::Error> {
            data: mk(),
            error: mk(),
        }
    })
}
