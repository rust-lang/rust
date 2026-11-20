//! Test documenting unusual behavior related to <https://github.com/rust-lang/rust/issues/161852>:
//! currently, fake reads for fake borrows on indexed-into slice pointers are emitted after bounds-
//! checks, to keep them alive for those. This does not keep fake borrows alive at points from which
//! all further bounds-checks are unreachable.
//@ check-pass

fn main() {
    let mut x: &[&[&[u32]]] = &[&[&[0]]];
    let y: &[&[&[u32]]] = &[];
    x[0][{ x = y; return; 0 }];
}
