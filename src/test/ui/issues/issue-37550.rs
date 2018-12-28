const fn x() {
    let t = true;
    let x = || t; //~ ERROR function pointers in const fn are unstable
}

fn main() {}
