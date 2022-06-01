// compile-flags: -Zmiri-permissive-provenance -Zmiri-disable-stacked-borrows -Zmiri-allow-ptr-int-transmute

fn main() {
    let x: i32 = 3;
    let x_ptr = &x as *const i32;

    // TODO: switch this to addr() once we intrinsify it
    let x_usize: usize = unsafe { std::mem::transmute(x_ptr) };
    // Cast back a pointer that did *not* get exposed.
    let ptr = x_usize as *const i32;
    assert_eq!(unsafe { *ptr }, 3); //~ ERROR Undefined Behavior: dereferencing pointer failed
}
