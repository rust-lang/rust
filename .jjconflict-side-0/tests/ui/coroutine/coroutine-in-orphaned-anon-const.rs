//@ edition:2021

trait X {
    fn test() -> Self::Assoc<{ async {} }>;
    //~^ ERROR associated type `Assoc` not found for `Self`
}

pub fn main() {}
