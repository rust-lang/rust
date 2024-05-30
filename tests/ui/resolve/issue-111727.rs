//@ edition: 2021

fn main() {
    std::any::Any::create();
    //~^ ERROR trait objects must include the `dyn` keyword
}
