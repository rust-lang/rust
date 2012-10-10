fn main() {
    let arr = [1,2,3];
    let struc = {a: 13u8, b: arr, c: 42};
    let s = sys::log_str(&struc);
    assert(s == ~"{ a: 13, b: [ 1, 2, 3 ], c: 42 }");
}
