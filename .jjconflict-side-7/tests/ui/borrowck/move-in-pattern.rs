//@ run-rustfix
// Issue #63988
#[derive(Debug)]
struct S;
fn foo(_: Option<S>) {}

enum E {
    V {
        s: S,
    }
}
fn bar(_: E) {}

fn main() {
    let s = Some(S);
    if let Some(x) = s {
        let _ = x;
    }
    foo(s); //~ ERROR use of partially moved value: `s`
    let e = E::V { s: S };
    let E::V { s: x } = e;
    let _ = x;
    bar(e); //~ ERROR use of partially moved value: `e`
}
