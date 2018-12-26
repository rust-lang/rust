struct T1 { //~ ERROR E0072
    foo: isize,
    foolish: T1
}

fn main() { }
