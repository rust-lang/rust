trait Trait {
    fn foo();
}

fn main() {
    <<i32 as Copy>::foobar as Trait>::foo();
    //~^ ERROR cannot find associated type `foobar` in trait `Copy`
}
