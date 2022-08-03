// check-pass

pub trait Foo {
    type Bar;
}

pub trait Broken {
    type Assoc;
    fn broken(&self) where Self::Assoc: Foo;
}

impl<T> Broken for T {
    type Assoc = ();
    fn broken(&self) where Self::Assoc: Foo {
        //~^ WARNING the where-clause bound
        let _x: <Self::Assoc as Foo>::Bar;
    }
}

fn main() {
    let _m: &dyn Broken<Assoc=()> = &();
}
