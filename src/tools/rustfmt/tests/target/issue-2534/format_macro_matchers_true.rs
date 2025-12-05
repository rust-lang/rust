// rustfmt-format_macro_matchers: true

macro_rules! foo {
    ($a:ident : $b:ty) => {};
    ($a:ident $b:ident $c:ident) => {};
}
