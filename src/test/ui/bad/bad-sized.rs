// ignore-tidy-linelength

trait Trait {}

pub fn main() {
    let x: Vec<dyn Trait + Sized> = Vec::new();
    //~^ ERROR only auto traits can be used as additional traits in a trait object [E0225]
    //~| ERROR the size for values of type `dyn std::marker::Sized` cannot be known at compilation time [E0277]
    //~| the trait `std::marker::Sized` cannot be made into an object [E0038]
    //~| ERROR the size for values of type `dyn std::marker::Sized` cannot be known at compilation time [E0277]
    //~| the trait `std::marker::Sized` cannot be made into an object [E0038]
}
