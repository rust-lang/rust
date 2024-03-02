#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

// Check that we allow & respect trailing where-clauses on lazy type aliases.

type Alias<T> = T
where
    String: From<T>;

fn main() {
    let _: Alias<&str>;
    let _: Alias<()>; //~ ERROR trait `From<()>` is not implemented for `String`
}
