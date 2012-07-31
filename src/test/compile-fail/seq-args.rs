use std;
fn main() {
trait seq { }

impl <T> of seq<T> for ~[T] { //~ ERROR wrong number of type arguments
    /* ... */
}
impl of seq<bool> for u32 {
   /* Treat the integer as a sequence of bits */
}

}
