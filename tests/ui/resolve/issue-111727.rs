//@ edition: 2021

fn main() {
    std::any::Any::create();
    //~^ ERROR trait objects must include the `dyn` keyword
    //~| ERROR no function or associated item named `create` found for trait `Any`
}
