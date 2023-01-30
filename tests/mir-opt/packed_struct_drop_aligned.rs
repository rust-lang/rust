// ignore-wasm32-bare compiled with panic=abort by default


// EMIT_MIR packed_struct_drop_aligned.main.SimplifyCfg-elaborate-drops.after.mir
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
