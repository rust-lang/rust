pub static V: &u32 = &X;
pub static F: fn() = f;
pub static G: fn() = G0;

static X: u32 = 42;
static G0: fn() = g;

pub fn v() -> *const u32 {
    V
}

fn f() {}

fn g() {}
