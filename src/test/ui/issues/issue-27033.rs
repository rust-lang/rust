// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl
fn main() {
    match Some(1) {
        None @ _ => {} //~ ERROR match bindings cannot shadow unit variants
    };
    const C: u8 = 1;
    match 1 {
        C @ 2 => { //~ ERROR match bindings cannot shadow constant
            println!("{}", C);
        }
        _ => {}
    };
}
