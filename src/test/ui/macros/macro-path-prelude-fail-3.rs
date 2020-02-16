// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl
fn main() {
    inline!(); //~ ERROR cannot find macro `inline` in this scope
}
