//@ run-pass
// Regression test for #49685: drop elaboration was not revealing the
// value of `impl Trait` returns, leading to an ICE.

fn main() {
    let _ = Some(())
        .into_iter()
        .flat_map(|_| Some(()).into_iter().flat_map(func));
}

fn func(_: ()) -> impl Iterator<Item = ()> {
    Some(()).into_iter().flat_map(|_| vec![])
}
