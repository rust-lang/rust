#[macro_export]
macro_rules! helper {
    // Use `:tt` instead of `:ident` so that we don't get a `None`-delimited group
    ($first:tt) => {
        pub fn foo<T>() {
            // The span of `$first` comes from another file,
            // so the expression `1 + $first` ends up with an
            // 'invalid' span that starts and ends in different files.
            // We use the `respan!` macro to give all tokens the same
            // `SyntaxContext`, so that the parser will try to merge the spans.
            respan::respan!(let a = 1 + $first;);
        }
    }
}
