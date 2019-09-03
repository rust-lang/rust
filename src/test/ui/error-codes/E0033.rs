trait SomeTrait {
    fn foo();
}

fn main() {
    let trait_obj: &dyn SomeTrait = SomeTrait;
    //~^ ERROR expected value, found trait `SomeTrait`
    //~| ERROR E0038
    //~| associated function `foo` has no `self` parameter

    let &invalid = trait_obj;
    //~^ ERROR E0033
}
