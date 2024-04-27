// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

// Ensure that there are no drop terminators in `unwrap<T>` (except the one along the cleanup
// path).

// EMIT_MIR no_drop_for_inactive_variant.unwrap.SimplifyCfg-pre-optimizations.after.mir
fn unwrap<T>(opt: Option<T>) -> T {
    match opt {
        Some(x) => x,
        None => panic!(),
    }
}

fn main() {
    let _ = unwrap(Some(1i32));
}
