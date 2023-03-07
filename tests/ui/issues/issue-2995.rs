fn bad (p: *const isize) {
    let _q: &isize = p as &isize; //~ ERROR non-primitive cast
}

fn main() { }
