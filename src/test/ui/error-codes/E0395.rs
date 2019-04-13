// gate-test-const_compare_raw_pointers

static FOO: i32 = 42;
static BAR: i32 = 42;

static BAZ: bool = unsafe { (&FOO as *const i32) == (&BAR as *const i32) };
//~^ ERROR comparing raw pointers inside static

fn main() {
}
