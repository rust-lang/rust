#![feature(checked_type_aliases)]

// Check that we allow & respect trailing where-clauses on checked type aliases.

type Alias<T> = T
where
    String: From<T>;

fn main() {
    let _: Alias<&str>;
    let _: Alias<()>; //~ ERROR the trait bound `String: From<()>` is not satisfied
}
