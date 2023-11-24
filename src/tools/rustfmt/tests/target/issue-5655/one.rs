// rustfmt-version: One

fn foo<T>(_: T)
where
    T: std::fmt::Debug,

    T: std::fmt::Display,
{
}
