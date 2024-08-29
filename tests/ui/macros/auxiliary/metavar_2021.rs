//@ edition: 2021
#[macro_export]
macro_rules! make_matcher {
    ($name:ident, $fragment_type:ident, $d:tt) => {
        #[macro_export]
        macro_rules! $name {
            ($d _:$fragment_type) => { true };
            (const { 0 }) => { false };
        }
    };
}
make_matcher!(is_expr_from_2021, expr, $);
