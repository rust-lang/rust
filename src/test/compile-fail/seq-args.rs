extern mod std;
fn main() {
trait seq { }

impl<T> ~[T]: seq<T> { //~ ERROR wrong number of type arguments
    /* ... */
}
impl u32: seq<bool> {
   /* Treat the integer as a sequence of bits */
}

}
