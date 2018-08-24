fn main() {
trait seq { }

impl<T> seq<T> for Vec<T> { //~ ERROR wrong number of type arguments
    /* ... */
}
impl seq<bool> for u32 { //~ ERROR wrong number of type arguments
   /* Treat the integer as a sequence of bits */
}

}
