

iter range(lo: uint, hi: uint) -> uint {
    let lo_ = lo;
    while lo_ < hi { put lo_; lo_ += 1u; }
}

fn create_index<T>(index: &[{a: T, b: uint}], hash_fn: fn(&T) -> uint ) {
    for each i: uint in range(0u, 256u) { let bucket: [T] = ~[]; }
}

fn main() { }
