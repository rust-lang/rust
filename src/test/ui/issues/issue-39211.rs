#![feature(associated_consts)]

trait VecN {
    const DIM: usize;
}
trait Mat {
    type Row: VecN;
}

fn m<M: Mat>() {
    let a = [3; M::Row::DIM]; //~ ERROR associated type `Row` not found for `M`
}
fn main() {
}
