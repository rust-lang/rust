fn func(i: i32) {
    i(); //~ERROR expected function, found `i32`
}
fn main() {
    let i = 0i32;
    i(); //~ERROR expected function, found `i32`
}
