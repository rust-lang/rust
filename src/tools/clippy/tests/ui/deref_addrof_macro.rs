macro_rules! m {
    ($($x:tt),*) => { &[$(($x, stringify!(x)),)*] };
}

#[warn(clippy::deref_addrof)]
fn f() -> [(i32, &'static str); 3] {
    *m![1, 2, 3] // should be fine
}

fn main() {}
