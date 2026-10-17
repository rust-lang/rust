//@ edition: 2021

fn main() {
    std::any::Any::create();
    //~^ ERROR expected a type, found a trait
}
