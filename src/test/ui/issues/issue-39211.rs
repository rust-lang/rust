#![feature(associated_consts)]

trait VecN {
    const DIM: usize;
}
trait Mat {
    type Row: VecN;
}

fn m<M: Mat>() {
    let a = [3; M::Row::DIM];
    //~^ ERROR type parameters can't appear within an array length expression [E0447]
}
fn main() {
}
