// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl

fn main() {
    let xs : Vec<Option<i32>> = vec![Some(1), None];

    for Some(x) in xs {}
    //~^ ERROR E0005
}
