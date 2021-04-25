pub static V: &u32 = &X;
pub static F: fn() = f;

static X: u32 = 42;

pub fn v() -> *const u32 {
    V
}

fn f() {}
