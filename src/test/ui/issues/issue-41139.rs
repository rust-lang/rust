trait Trait {}

fn get_function<'a>() -> &'a Fn() -> Trait { panic!("") }

fn main() {
    let t : &Trait = &get_function()();
    //~^ ERROR cannot move a value of type dyn Trait
}
