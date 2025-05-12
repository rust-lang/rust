mod banana {
    //~^ HELP the following traits which provide `pick` are implemented but not in scope
    pub struct Chaenomeles;

    pub trait Apple {
        fn pick(&self) {}
    }
    impl Apple for Chaenomeles {}

    pub trait Peach {
        fn pick(&self, a: &mut ()) {}
    }
    impl<Mango: Peach> Peach for Box<Mango> {}
    impl Peach for Chaenomeles {}
}

fn main() {
    banana::Chaenomeles.pick()
    //~^ ERROR no method named
    //~| HELP items from traits can only be used if the trait is in scope
}
