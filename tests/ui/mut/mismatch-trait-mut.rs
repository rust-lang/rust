// This test ensures that the suggested path is actually valid and relative to the
// trait impl and not to the trait definition.

//@ run-rustfix

mod bar {
    #[allow(dead_code)]
    pub struct X;
}

pub trait A {
    fn f(_x: &mut bar::X) {}
    //~^ NOTE type in trait
    fn g<'a>(_x: &'a mut bar::X) -> &'a () { &() }
    //~^ NOTE type in trait
    fn f2(_x: &bar::X) {}
    //~^ NOTE type in trait
    fn g2<'a>(_x: &'a bar::X) -> &'a () { &() }
    //~^ NOTE type in trait
}

mod b {
    #[allow(dead_code)]
    pub struct X;

    impl crate::A for X {
        fn f(_x: &crate::bar::X) {}
        //~^ ERROR method `f` has an incompatible type for trait
        //~| NOTE types differ in mutability
        //~| NOTE expected signature `fn(&mut X)`
        fn g<'a>(_x: &'a crate::bar::X) -> &'a () { &() }
        //~^ ERROR method `g` has an incompatible type for trait
        //~| NOTE types differ in mutability
        //~| NOTE expected signature `fn(&'a mut X) -> &'a ()`
        fn f2(_x: &mut crate::bar::X) {}
        //~^ ERROR method `f2` has an incompatible type for trait
        //~| NOTE types differ in mutability
        //~| NOTE expected signature `fn(&X)`
        fn g2<'a>(_x: &'a mut crate::bar::X) -> &'a () { &() }
        //~^ ERROR method `g2` has an incompatible type for trait
        //~| NOTE types differ in mutability
        //~| NOTE expected signature `fn(&'a X) -> &'a ()`
    }
}

fn main() {}
