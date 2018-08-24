// aux-build:linkage-visibility.rs
// ignore-android: FIXME(#10356)
// ignore-windows: std::dynamic_lib does not work on Windows well
// ignore-emscripten no dynamic linking

extern crate linkage_visibility as foo;

pub fn main() {
    foo::test();
    foo::foo2::<isize>();
    foo::foo();
}
