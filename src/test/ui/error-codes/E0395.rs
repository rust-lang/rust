static FOO: i32 = 42;
static BAR: i32 = 42;

static BAZ: bool = unsafe { (&FOO as *const i32) == (&BAR as *const i32) };
//~^ ERROR pointers cannot be compared in a meaningful way during const eval

fn main() {
}
