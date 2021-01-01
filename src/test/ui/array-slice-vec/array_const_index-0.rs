const A: &'static [i32] = &[];
const B: i32 = (&A)[1];
//~^ index out of bounds: the length is 0 but the index is 1
//~| ERROR any use of this value will cause an error

fn main() {
    let _ = B;
}
