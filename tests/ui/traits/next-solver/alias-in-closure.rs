//@compile-flags: -Znext-solver

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
