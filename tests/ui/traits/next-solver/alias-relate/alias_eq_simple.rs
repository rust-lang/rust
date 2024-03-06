//@ check-pass
//@ compile-flags: -Znext-solver

// test that the new solver can handle `alias-eq(<i32 as TraitB>::Assoc, u32)`

trait TraitA {}

trait TraitB {
    type Assoc;
}

impl<T: TraitB> TraitA for (T, T::Assoc) {}

impl TraitB for i32 {
    type Assoc = u32;
}

fn needs_a<T: TraitA>() {}

fn main() {
    needs_a::<(i32, u32)>();
}
