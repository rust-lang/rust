//@ compile-flags: -Znext-solver
//@ check-pass

// A recursive non-self alias bound should not make an otherwise
// provable goal ambiguous. See trait-system-refactor-initiative#313.

trait Inner {
    type This: Outer<Assoc: Trait<IfNormToSelf: Trait>>;
}

trait Outer {
    type Assoc;
}

trait Trait {
    type IfNormToSelf;
}

fn main() {}
