//@ edition: 2024

// This contains a binding in edition 2024, so if matched with a reference binding mode it will end
// up with a `mut ref mut` binding mode. We use this to test the migration lint on patterns with
// mixed editions.
#[macro_export]
macro_rules! mixed_edition_pat {
    ($foo:ident) => {
        Some(mut $foo)
    };
}

#[macro_export]
macro_rules! bind_ref {
    ($foo:ident) => {
        ref $foo
    };
}
