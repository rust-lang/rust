#[repr(transparent)]
struct Wrap([u8; 0]);

fn callee(_s: Wrap) {}
//~^ ERROR: type Wrap passing argument of type ()

fn main() {
    let fnptr: fn(Wrap) = callee;
    let fnptr: fn(()) = unsafe { std::mem::transmute(fnptr) };
    fnptr(());
}
