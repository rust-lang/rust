trait Trait {}

fn get_function<'a>() -> &'a dyn Fn() -> dyn Trait {
    panic!("")
}

fn main() {
    // This isn't great. The issue here is that `dyn Trait` is not sized, so
    // `dyn Fn() -> dyn Trait` is not well-formed.
    let t: &dyn Trait = &get_function()();
    //~^ ERROR expected function, found `&dyn Fn() -> (dyn Trait + 'static)`
}
