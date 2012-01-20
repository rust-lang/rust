// error-pattern: unresolved name

// Tag variants are not exported with their tags. This allows for a
// simple sort of ADT.

mod foo {
    export t;

    enum t { t1, }
}

fn main() { let x = foo::t1; }
