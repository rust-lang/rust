//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl)]

#[const_trait]
trait Trait {
    fn method();
}

const fn foo<T: Trait>() {
    let _ = || {
        // Make sure this doesn't enforce `T: [const] Trait`
        T::method();
    };
}

fn bar<T: const Trait>() {
    let _ = || {
        // Make sure unconditionally const bounds propagate from parent.
        const {
            T::method();
        };
    };
}

fn main() {}
