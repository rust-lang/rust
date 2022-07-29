static FOO: i32 = 42;
static BAR: i32 = 42;

static BAZ: bool = unsafe { (&FOO as *const i32) == (&BAR as *const i32) };
//~^ ERROR pointers cannot be reliably compared during const eval

fn main() {
}
