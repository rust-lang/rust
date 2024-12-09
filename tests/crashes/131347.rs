//@ known-bug: #131347
//@ compile-flags: -Zmir-enable-passes=+GVN -Zmir-enable-passes=+Inline -Zvalidate-mir

struct S;
static STUFF: [i8] = [0; S::N];

fn main() {
    assert_eq!(STUFF, [0; 63]);
}
