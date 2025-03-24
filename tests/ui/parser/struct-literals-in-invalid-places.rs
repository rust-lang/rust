fn main() {
    if Foo { x: 3 }.hi() { //~ ERROR struct literals are not allowed here
        println!("yo");
    }
    if let true = Foo { x: 3 }.hi() { //~ ERROR struct literals are not allowed here
        println!("yo");
    }

    for x in Foo { x: 3 }.hi() { //~ ERROR struct literals are not allowed here
        //~^ ERROR `bool` is not an iterator
        println!("yo");
    }

    while Foo { x: 3 }.hi() { //~ ERROR struct literals are not allowed here
        println!("yo");
    }
    while let true = Foo { x: 3 }.hi() { //~ ERROR struct literals are not allowed here
        println!("yo");
    }

    match Foo { x: 3 } { //~ ERROR struct literals are not allowed here
        Foo { x: x } => {}
    }

    let _ = |x: E| {
        let field = true;
        if x == E::V { field } {}
        //~^ ERROR expected value, found struct variant `E::V`
        //~| ERROR mismatched types
        if x == E::I { field1: true, field2: 42 } {}
        //~^ ERROR struct literals are not allowed here
        if x == E::V { field: false } {}
        //~^ ERROR struct literals are not allowed here
        if x == E::J { field: -42 } {}
        //~^ ERROR struct literals are not allowed here
        if x == E::K { field: "" } {}
        //~^ ERROR struct literals are not allowed here
        let y: usize = ();
        //~^ ERROR mismatched types
    };

    // Regression test for <https://github.com/rust-lang/rust/issues/43412>.
    while || Foo { x: 3 }.hi() { //~ ERROR struct literals are not allowed here
        //~^ ERROR mismatched types
        println!("yo");
    }

    // Regression test for <https://github.com/rust-lang/rust/issues/82051>.
    if Foo { x: one(), }.hi() { //~ ERROR struct literals are not allowed here
        println!("Positive!");
    }
}

struct Foo {
    x: isize,
}

impl Foo {
    fn hi(&self) -> bool {
        true
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
enum E {
    V { field: bool },
    I { field1: bool, field2: usize },
    J { field: isize },
    K { field: &'static str},
}

fn one() -> isize { 1 }
