fn foo(cond: bool) {
    let x = 5;
    let mut y: &blk.int = &x;

    let mut z: &blk.int;
    if cond {
        z = &x;
    } else {
        let w: &blk.int = &x;
        z = w; //! ERROR mismatched types
    }
}

fn main() {
}