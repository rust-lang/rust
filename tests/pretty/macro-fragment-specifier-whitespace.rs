//@ pretty-mode:expanded
//@ pp-exact:macro-fragment-specifier-whitespace.pp

// Test that fragment specifier names in macro definitions are properly
// separated from the following keyword/identifier token when pretty-printed.
// This is a regression test for a bug where `$x:ident` followed by `where`
// was pretty-printed as `$x:identwhere` (an invalid fragment specifier).

macro_rules! outer {
    ($d:tt $($params:tt)*) => {
        #[macro_export]
        macro_rules! inner {
            ($($params)* where $d($rest:tt)*) => {};
        }
    };
}
outer!($ $x:ident);

fn main() {}
