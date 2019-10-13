trait SomeTrait {
    fn foo(); //~ associated function `foo` has no `self` parameter
}

fn main() {
    let trait_obj: &dyn SomeTrait = SomeTrait;
    //~^ ERROR expected value, found trait `SomeTrait`
    //~| ERROR E0038

    let &invalid = trait_obj;
    //~^ ERROR E0033
}
