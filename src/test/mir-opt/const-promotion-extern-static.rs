extern "C" {
    static X: i32;
}

static Y: i32 = 42;

// EMIT_MIR rustc.BAR.PromoteTemps.diff
// EMIT_MIR rustc.BAR-promoted[0].ConstProp.after.mir
static mut BAR: *const &i32 = [&Y].as_ptr();

// EMIT_MIR rustc.FOO.PromoteTemps.diff
// EMIT_MIR rustc.FOO-promoted[0].ConstProp.after.mir
static mut FOO: *const &i32 = [unsafe { &X }].as_ptr();

fn main() {}
