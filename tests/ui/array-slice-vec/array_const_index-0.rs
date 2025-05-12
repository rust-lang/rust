const A: &'static [i32] = &[];
const B: i32 = (&A)[1];
//~^ NOTE index out of bounds: the length is 0 but the index is 1
//~| ERROR evaluation of constant value failed

fn main() {
    let _ = B;
}
