struct MyS;

impl MyS {
    const FOO: i32 = 1;
    fn foo() -> MyS {
        MyS
    }
}

fn main() {
    let x: i32 = MyS::foo;
    //~^ ERROR mismatched types
    //~| HELP try referring to the

    let z: i32 = i32::max;
    //~^ ERROR mismatched types
    //~| HELP try referring to the

    // This example is still broken though... This is a hard suggestion to make,
    // because we don't have access to the associated const probing code to make
    // this suggestion where it's emitted, i.e. in trait selection.
    let y: i32 = i32::max - 42;
    //~^ ERROR cannot subtract
    //~| HELP use parentheses
}
