// error-pattern:expected function or native function but found *u8
crust fn f() {
}

fn main() {
    let x = bind f();
}