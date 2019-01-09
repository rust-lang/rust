fn main() {
trait Seq { }

impl<T> Seq<T> for Vec<T> { //~ ERROR wrong number of type arguments
    /* ... */
}
impl Seq<bool> for u32 { //~ ERROR wrong number of type arguments
   /* Treat the integer as a sequence of bits */
}

}
