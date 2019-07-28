// run-pass
// ignore-emscripten

#[repr(C)]
pub struct Foo(i128);

#[no_mangle]
pub extern "C" fn foo(x: Foo) -> Foo { x }

fn main() {
    foo(Foo(1));
}
