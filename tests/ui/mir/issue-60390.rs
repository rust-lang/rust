//@ check-pass
//@ compile-flags: --emit=mir,link
// Regression test for #60390, this ICE requires `--emit=mir` flag.

fn main() {
    enum Inner { Member(u32) };
    Inner::Member(0);
}
