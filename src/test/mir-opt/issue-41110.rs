// ignore-wasm32-bare compiled with panic=abort by default

// check that we don't emit multiple drop flags when they are not needed.


// EMIT_MIR rustc.main.ElaborateDrops.after.mir
fn main() {
    let x = S.other(S.id());
}

// no_mangle to make sure this gets instantiated even in an executable.
#[no_mangle]
// EMIT_MIR rustc.test.ElaborateDrops.after.mir
pub fn test() {
    let u = S;
    let mut v = S;
    drop(v);
    v = u;
}

struct S;
impl Drop for S {
    fn drop(&mut self) {
    }
}

impl S {
    fn id(self) -> Self { self }
    fn other(self, s: Self) {}
}
