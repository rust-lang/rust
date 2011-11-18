// Issue #976

fn f<copy T>(x: ~T) {
    let _x2 = x;
}
fn main() { }
