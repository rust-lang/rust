//@ run-pass
//@ needs-threads

#[repr(C)]
pub struct Foo(i128);

#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn foo(x: Foo) -> Foo { x }

fn main() {
    foo(Foo(1));
}
