trait VecN {
    const DIM: usize;
}
trait Mat {
    type Row: VecN;
}

fn m<M: Mat>() {
    let a = [3; M::Row::DIM];
    //~^ ERROR constant expression depends on a generic parameter
    //~| ERROR constant expression depends on a generic parameter
}
fn main() {
}

// https://github.com/rust-lang/rust/issues/39211
