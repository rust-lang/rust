//! Check for compilation errors when a trait is used with an incorrect number of generic arguments.

fn main() {
    trait Seq {}

    impl<T> Seq<T> for Vec<T> {
        //~^ ERROR trait takes 0 generic arguments but 1 generic argument
        /* ... */
    }

    impl Seq<bool> for u32 {
        //~^ ERROR trait takes 0 generic arguments but 1 generic argument
        /* Treat the integer as a sequence of bits */
    }
}
