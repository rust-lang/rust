struct Struct;
//~^ NOTE function or associated item `fob` not found for this struct

impl Struct {
    fn foo() { }
}

mod module {
    fn foo() { }

    struct Struct;

    impl Struct {
        fn foo() { }
    }
}

trait Trait {
    fn foo();
}

fn main() {
    Struct::fob();
    //~^ ERROR no function or associated item named `fob` found for struct `Struct` in the current scope
    //~| NOTE function or associated item not found in `Struct`

    Struc::foo();
    //~^ ERROR cannot find item `Struc`
    //~| NOTE use of undeclared type `Struc`

    modul::foo();
    //~^ ERROR cannot find item `modul`
    //~| NOTE use of undeclared crate or module `modul`

    module::Struc::foo();
    //~^ ERROR cannot find item `Struc`
    //~| NOTE could not find `Struc` in `module`

    Trai::foo();
    //~^ ERROR cannot find item `Trai`
    //~| NOTE use of undeclared type `Trai`
}
