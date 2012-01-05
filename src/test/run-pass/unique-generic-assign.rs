// Issue #976

fn f<T: copy>(x: ~T) {
    let _x2 = x;
}
fn main() { }
