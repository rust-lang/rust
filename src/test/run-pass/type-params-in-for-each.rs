

fn range(lo: uint, hi: uint, it: block(uint)) {
    let lo_ = lo;
    while lo_ < hi { it(lo_); lo_ += 1u; }
}

fn create_index<@T>(index: [{a: T, b: uint}], hash_fn: fn(T) -> uint) {
    range(0u, 256u) {|_i| let bucket: [T] = []; };
}

fn main() { }
