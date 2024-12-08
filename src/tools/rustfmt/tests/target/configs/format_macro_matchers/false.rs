// rustfmt-format_macro_matchers: false

macro_rules! foo {
    ($a: ident : $b: ty) => { $a(42): $b; };
    ($a: ident $b: ident $c: ident) => { $a=$b+$c; };
}
