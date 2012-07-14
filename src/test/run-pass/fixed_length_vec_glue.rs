fn main() {
    let arr = [1,2,3]/3;
    let struct = {a: 13u8, b: arr, c: 42};
    let s = sys::log_str(struct);
    assert(s == ~"(13, [1, 2, 3]/3, 42)");
}
