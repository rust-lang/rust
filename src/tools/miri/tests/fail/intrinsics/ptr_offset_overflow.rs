//@normalize-stderr-test: "\d+ bytes" -> "$$BYTES bytes"

fn main() {
    let v = [0i8; 4];
    let x = &v as *const i8;
    let x = unsafe { x.offset(isize::MIN) }; //~ERROR: out-of-bounds pointer arithmetic
    panic!("this should never print: {:?}", x);
}
