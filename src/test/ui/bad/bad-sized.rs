trait Trait {}

pub fn main() {
    let x: Vec<dyn Trait + Sized> = Vec::new();
    //~^ ERROR only auto traits can be used as additional traits in a trait object
    //~| ERROR the size for values of type
    //~| ERROR the size for values of type
}
