//@ edition: 2021

fn main() {
    std::any::Any::create();
    //~^ ERROR no function or associated item named `create` found for trait `Any`
    //~| ERROR trait objects must include the `dyn` keyword
}
