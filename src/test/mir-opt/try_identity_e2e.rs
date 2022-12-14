// Track the status of MIR optimizations simplifying `Ok(res?)` for both the old and new desugarings
// of that syntax.

use std::ops::ControlFlow;

// EMIT_MIR try_identity_e2e.new.PreCodegen.after.mir
fn new<T, E>(x: Result<T, E>) -> Result<T, E> {
    Ok(
        match {
            match x {
                Ok(v) => ControlFlow::Continue(v),
                Err(e) => ControlFlow::Break(e),
            }
        } {
            ControlFlow::Continue(v) => v,
            ControlFlow::Break(e) => return Err(e),
        }
    )
}

// EMIT_MIR try_identity_e2e.old.PreCodegen.after.mir
fn old<T, E>(x: Result<T, E>) -> Result<T, E> {
    Ok(
        match x {
            Ok(v) => v,
            Err(e) => return Err(e),
        }
    )
}

fn main() {
    let _ = new::<(), ()>(Ok(()));
    let _ = old::<(), ()>(Ok(()));
}
