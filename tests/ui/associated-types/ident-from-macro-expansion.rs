trait Trait {}
impl Trait for () {}

macro_rules! fully_qualified {
    ($id:ident) => {
        <() as Trait>::$id
    }
}

macro_rules! type_dependent {
    ($t:ident, $id:ident) => {
        T::$id
    }
}

fn t<T: Trait>() {
    let x: fully_qualified!(Assoc);
    //~^ ERROR cannot find associated type `Assoc` in trait `Trait`
    let x: type_dependent!(T, Assoc);
    //~^ ERROR associated type `Assoc` not found for `T`
}

fn main() {}
