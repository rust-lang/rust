//@normalize-stderr-test: "\d+ bytes" -> "$$BYTES bytes"

fn main() {
    let v = [0i8; 4];
    let x = &v as *const i8;
    let x = unsafe { x.offset(isize::MIN) }; //~ERROR: in-bounds pointer arithmetic failed
    panic!("this should never print: {:?}", x);
}
