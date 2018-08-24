static a: &'static str = "foo";
static b: *const u8 = a as *const u8; //~ ERROR casting
static c: *const u8 = &a as *const u8; //~ ERROR casting

fn main() {
}
