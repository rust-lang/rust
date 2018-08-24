const A: &'static [i32] = &[];
const B: i32 = (&A)[1];
//~^ index out of bounds: the len is 0 but the index is 1
//~| ERROR this constant cannot be used

fn main() {
    let _ = B;
}
