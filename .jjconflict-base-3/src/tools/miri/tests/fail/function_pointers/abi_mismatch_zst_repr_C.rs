#[repr(C)]
struct C;

fn callee() {}
//~^ ERROR: return type () passing return place of type C

fn main() {
    let fnptr: fn() -> () = callee;
    let fnptr: fn() -> C = unsafe { std::mem::transmute(fnptr) };
    fnptr();
}
