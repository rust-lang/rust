// compile-flags: -Ztrait-solver=next

trait Mirror {
    type Assoc: ?Sized;
}

struct Wrapper<T: ?Sized>(T);
impl<T: ?Sized> Mirror for Wrapper<T> {
    type Assoc = T;
}

fn mirror<W: Mirror>(_: W) -> Box<W::Assoc> { todo!() }

fn type_error() -> TypeError { todo!() }
//~^ ERROR cannot find type `TypeError` in this scope

fn main() {
    let x = mirror(type_error());
}
