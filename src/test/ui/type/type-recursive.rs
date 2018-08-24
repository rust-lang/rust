struct t1 { //~ ERROR E0072
    foo: isize,
    foolish: t1
}

fn main() { }
