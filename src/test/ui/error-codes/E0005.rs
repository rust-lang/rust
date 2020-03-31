// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl

fn main() {
    let x = Some(1);
    let Some(y) = x; //~ ERROR E0005
}
