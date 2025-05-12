//@ revisions: rpass1 rpass2
//@ edition: 2024

#![allow(unused)]

fn main() {
    #[cfg(rpass1)]
    async || {};

    #[cfg(rpass2)]
    || {
        || ();
        || ();
    };
}
