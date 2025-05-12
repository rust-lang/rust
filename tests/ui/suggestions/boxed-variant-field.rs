enum Ty {
    Unit,
    List(Box<Ty>),
}

fn foo(x: Ty) -> Ty {
    match x {
        Ty::Unit => Ty::Unit,
        Ty::List(elem) => foo(elem),
        //~^ ERROR mismatched types
        //~| HELP consider unboxing the value
    }
}

fn main() {}
