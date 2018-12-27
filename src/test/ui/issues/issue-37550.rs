const fn x() {
    let t = true; //~ ERROR local variables in const fn
    let x = || t;
}

fn main() {}
