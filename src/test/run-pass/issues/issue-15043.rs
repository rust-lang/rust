// run-pass
// pretty-expanded FIXME #23616

#![allow(warnings)]

struct S<T>(T);

static s1: S<S<usize>>=S(S(0));
static s2: S<usize>=S(0);

fn main() {
    let foo: S<S<usize>>=S(S(0));
    let foo: S<usize>=S(0);
}
