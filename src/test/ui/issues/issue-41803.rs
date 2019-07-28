// run-pass
/// A compile-time map from identifiers to arbitrary (heterogeneous) expressions
macro_rules! ident_map {
    ( $name:ident = { $($key:ident => $e:expr,)* } ) => {
        macro_rules! $name {
            $(
                ( $key ) => { $e };
            )*
            // Empty invocation expands to nothing. Needed when the map is empty.
            () => {};
        }
    };
}

ident_map!(my_map = {
    main => 0,
});

fn main() {
    my_map!(main);
}
