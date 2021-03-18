fn main() {
    trait Seq { }

    impl<T> Seq<T> for Vec<T> {
        //~^ ERROR this trait takes 0 type arguments but 1 type argument was supplied
        /* ... */
    }

    impl Seq<bool> for u32 {
        //~^ ERROR this trait takes 0 type arguments but 1 type argument was supplied
        /* Treat the integer as a sequence of bits */
    }
}
