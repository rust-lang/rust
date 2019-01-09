// Type arguments in unresolved entities (reporting errors before type checking)
// should have their types recorded.

trait Tr<T> {}

fn local_type() {
    let _: Nonexistent<u8, Assoc = u16>; //~ ERROR cannot find type `Nonexistent` in this scope
}

fn ufcs_trait() {
    <u8 as Tr<u8>>::nonexistent(); //~ ERROR cannot find method or associated constant `nonexistent`
}

fn ufcs_item() {
    NonExistent::Assoc::<u8>; //~ ERROR undeclared type or module `NonExistent`
}

fn method() {
    nonexistent.nonexistent::<u8>(); //~ ERROR cannot find value `nonexistent`
}

fn closure() {
    let _ = |a, b: _| -> _ { 0 }; // OK
}

fn main() {}
