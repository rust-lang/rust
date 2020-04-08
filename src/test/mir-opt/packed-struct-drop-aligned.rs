// ignore-wasm32-bare compiled with panic=abort by default

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR rustc.main.SimplifyCfg-elaborate-drops.after.mir
fn main() {
    let mut x = Packed(Aligned(Droppy(0)));
    x.0 = Aligned(Droppy(0));
}

struct Aligned(Droppy);
#[repr(packed)]
struct Packed(Aligned);

struct Droppy(usize);
impl Drop for Droppy {
    fn drop(&mut self) {}
}
