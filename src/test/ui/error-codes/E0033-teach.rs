// compile-flags: -Z teach

trait SomeTrait {
    fn foo();
}

fn main() {
    let trait_obj: &dyn SomeTrait = SomeTrait;
    //~^ ERROR expected value, found trait `SomeTrait`
    //~| ERROR E0038
    //~| method `foo` has no receiver

    let &invalid = trait_obj;
    //~^ ERROR E0033
}
