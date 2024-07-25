//@ check-pass

fn make<T: Indir0<Ty: Indir1<Ty = Struct>>>() {
    let _ = T::Ty::Ty { field: 0 };
}

trait Indir0 {
    type Ty: Indir1;
}

trait Indir1 {
    type Ty;
}

struct Struct {
    field: i32
}

fn main() {}
