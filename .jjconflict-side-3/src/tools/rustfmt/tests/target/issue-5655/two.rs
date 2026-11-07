// rustfmt-style_edition: 2024

fn foo<T>(_: T)
where
    T: std::fmt::Debug,
    T: std::fmt::Display,
{
}
