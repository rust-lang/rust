const a: &str = "foo";
const b: *const u8 = a as *const u8; //~ ERROR casting
const c: *const u8 = &a as *const u8; //~ ERROR casting

fn main() {
}
