// ignore-wasm32-bare compiled with panic=abort by default

// Ensure that there are no drop terminators in `unwrap<T>` (except the one along the cleanup
// path).

// EMIT_MIR rustc.unwrap.SimplifyCfg-elaborate-drops.after.mir
fn unwrap<T>(opt: Option<T>) -> T {
    match opt {
        Some(x) => x,
        None => panic!(),
    }
}

fn main() {
    let _ = unwrap(Some(1i32));
}
