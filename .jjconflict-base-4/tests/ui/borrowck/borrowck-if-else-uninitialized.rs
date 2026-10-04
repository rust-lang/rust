fn foo(b: bool) {
    let x;
    if b {
        x = 1;
    } else {
        println!("{x}"); //~ ERROR E0381
    }
}

fn main() {
    foo(true);
}
