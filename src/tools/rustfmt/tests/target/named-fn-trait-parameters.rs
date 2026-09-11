fn allowed<F>(
    data: &str,
    f1: impl Fn(msg: String),
    f2: impl Fn(_: String),
    f3: impl Fn(String, msg: String),
    f4: impl Fn(msg: String, String),
    fg: F,
) where
    F: Fn(msg: String),
{
}

my_macro!(f(x: &str));
my_macro!(f(x: &str) -> ());
my_macro!(g(n: i32, m: usize) -> usize);
