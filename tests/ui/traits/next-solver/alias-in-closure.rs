//@ compile-flags: -Znext-solver

// If we have an alias in a closure ty and the closure had errors during typeck,
// we need to make sure the alias is rigid when building the item's MIR to avoid
// ICEing.

trait Trait {
    type Assoc;
}

fn foo<T: Trait>() {
    let _ = |_: T::Assoc| {
        let Some(_) = Some(42);
        //~^ ERROR refutable pattern in local binding
    };
}

fn main() {}
