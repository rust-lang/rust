// gate-test-const_compare_raw_pointers

static FOO: i32 = 42;
static BAR: i32 = 42;

static BAZ: bool = unsafe { (&FOO as *const i32) == (&BAR as *const i32) }; //~ ERROR issue #53020
fn main() {
}
