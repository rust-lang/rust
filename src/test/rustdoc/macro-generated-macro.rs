macro_rules! outer {
    ($($matcher:tt)*) => {
        #[macro_export]
        macro_rules! inner {
            (<= $($matcher)* =>) => {};
        }
    }
}

// @has macro_generated_macro/macro.inner.html //pre 'macro_rules! inner {'
// @has - //pre '(<= type $($i : ident) :: * + $e : expr =>) => { ... };'
outer!(type $($i:ident)::* + $e:expr);

inner!(<= type foo::bar + x.sort() =>);
