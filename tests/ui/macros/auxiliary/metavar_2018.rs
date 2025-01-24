//@ edition: 2018
#[macro_export]
macro_rules! make_matcher {
    ($name:ident, $fragment_type:ident, $d:tt) => {
        #[macro_export]
        macro_rules! $name {
            ($d _:$fragment_type) => { true };
            (const { 0 }) => { false };
            (A | B) => { false };
        }
    };
}
make_matcher!(is_expr_from_2018, expr, $);
make_matcher!(is_pat_from_2018, pat, $);
