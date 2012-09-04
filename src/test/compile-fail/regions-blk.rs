fn foo(cond: bool) {
    let x = 5;
    let mut y: &blk/int = &x;

    let mut z: &blk/int;
    if cond {
        z = &x; //~ ERROR cannot infer an appropriate lifetime due to conflicting requirements
    } else {
        let w: &blk/int = &x;
        z = w;
    }
}

fn main() {
}
