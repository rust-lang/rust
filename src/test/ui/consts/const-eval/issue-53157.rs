// build-pass (FIXME(62277): could be check-pass?)

macro_rules! m {
    () => {{
        fn f(_: impl Sized) {}
        f
    }}
}

fn main() {
    fn f() -> impl Sized {};
    m!()(f());
}
