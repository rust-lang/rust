fn callee(_s: [u8; 0]) {}
//~^ ERROR: type [u8; 0] passing argument of type ()

fn main() {
    let fnptr: fn([u8; 0]) = callee;
    let fnptr: fn(()) = unsafe { std::mem::transmute(fnptr) };
    fnptr(());
}
