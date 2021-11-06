const fn x() {
    let t = true;
    let x = || t; //~ ERROR function pointer
}

fn main() {}
