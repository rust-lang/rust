// Issue #976

fn f<T: Copy>(x: ~T) {
    let _x2 = x;
}
fn main() { }
