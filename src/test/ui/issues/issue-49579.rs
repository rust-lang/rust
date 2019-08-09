// build-pass (FIXME(62277): could be check-pass?)
// ignore-emscripten no i128 support

fn fibs(n: u32) -> impl Iterator<Item=u128> {
    (0 .. n)
    .scan((0, 1), |st, _| {
        *st = (st.1, st.0 + st.1);
        Some(*st)
    })
    .map(&|(f, _)| f)
}

fn main() {
    println!("{:?}", fibs(10).collect::<Vec<_>>());
}
