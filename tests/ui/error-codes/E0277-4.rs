use std::fmt::Display;

trait ImplicitTrait {
    fn foo(&self);
}

trait ExplicitTrait {
    fn foo(&self);
}

trait DisplayTrait {
    fn foo(&self);
}

trait UnimplementedTrait {
    fn foo(&self);
}

// Implicitly requires `T: Sized`.
impl<T> ImplicitTrait for T {
    fn foo(&self) {}
}

// Explicitly requires `T: Sized`.
impl<T: Sized> ExplicitTrait for T {
    fn foo(&self) {}
}

// Requires `T: Display`.
impl<T: Display> DisplayTrait for T {
    fn foo(&self) {}
}

fn main() {
    // `[u8]` does not implement `Sized`.
    let x: &[u8] = &[];
    ImplicitTrait::foo(x);
    //~^ ERROR: the trait bound `[u8]: ImplicitTrait` is not satisfied [E0277]
    ExplicitTrait::foo(x);
    //~^ ERROR: the trait bound `[u8]: ExplicitTrait` is not satisfied [E0277]

    // `UnimplementedTrait` has no implementations.
    UnimplementedTrait::foo(x);
    //~^ ERROR: the trait bound `[u8]: UnimplementedTrait` is not satisfied [E0277]

    // `[u8; 0]` implements `Sized` but not `Display`.
    let x: &[u8; 0] = &[];
    DisplayTrait::foo(x);
    //~^ ERROR: the trait bound `[u8; 0]: DisplayTrait` is not satisfied [E0277]
}
