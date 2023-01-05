mod mul1 {
    pub trait Mul {}
}

mod mul2 {
    pub trait Mul {}
}

mod mul3 {
    enum Mul {
      Yes,
      No
    }
}

mod mul4 {
    type Mul = String;
}

mod mul5 {
    struct Mul{
        left_term: u32,
        right_term: u32
    }
}

#[derive(Debug)]
struct Foo;

// When we comment the next line:
//use mul1::Mul;

// BEFORE, we got the following error for the `impl` below:
//   error: use of undeclared trait name `Mul` [E0405]
// AFTER, we get this message:
//   error: trait `Mul` is not in scope.
//   help: ...
//   help: you can import several candidates into scope (`use ...;`):
//   help:   `mul1::Mul`
//   help:   `mul2::Mul`
//   help:   `std::ops::Mul`

impl Mul for Foo {
//~^ ERROR cannot find trait `Mul`
}

// BEFORE, we got:
//   error: use of undeclared type name `Mul` [E0412]
// AFTER, we get:
//   error: type name `Mul` is not in scope. Maybe you meant:
//   help: ...
//   help: you can import several candidates into scope (`use ...;`):
//   help:   `mul1::Mul`
//   help:   `mul2::Mul`
//   help:   `mul3::Mul`
//   help:   `mul4::Mul`
//   help:   and 2 other candidates
fn getMul() -> Mul {
//~^ ERROR cannot find type `Mul`
}

// Let's also test what happens if the trait doesn't exist:
impl ThisTraitReallyDoesntExistInAnyModuleReally for Foo {
//~^ ERROR cannot find trait `ThisTraitReallyDoesntExistInAnyModuleReally`
}

// Let's also test what happens if there's just one alternative:
impl Div for Foo {
//~^ ERROR cannot find trait `Div`
}

fn main() {
    let foo = Foo();
    println!("Hello, {:?}!", foo);
}
