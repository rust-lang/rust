trait Trait {}

fn get_function<'a>() -> &'a dyn Fn() -> dyn Trait { panic!("") }

fn main() {
    let t : &dyn Trait = &get_function()();
    //~^ ERROR cannot move a value of type dyn Trait
}
