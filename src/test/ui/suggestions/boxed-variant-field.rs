enum Ty {
    Unit,
    List(Box<Ty>),
}

fn foo(x: Ty) -> Ty {
    match x {
        Ty::Unit => Ty::Unit,
        Ty::List(elem) => foo(elem),
        //~^ ERROR mismatched types
        //~| HELP try dereferencing the `Box`
        //~| HELP try using a variant of the expected enum
    }
}

fn main() {}
