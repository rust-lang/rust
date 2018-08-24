trait VecN {
    const DIM: usize;
}

trait Mat {
    type Row: VecN;
}

fn m<M: Mat>() {
    let x = M::Row::DIM;
}

fn main() {}
