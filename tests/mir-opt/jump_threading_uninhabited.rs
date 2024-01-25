// unit-test: JumpThreading
// compile-flags: -Zmir-opt-level=3 -Zunsound-mir-opts
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// skip-filecheck

pub enum HiddenType {}

pub struct Wrap<T>(T);

// EMIT_MIR jump_threading_uninhabited.test_questionmark.JumpThreading.diff
fn test_questionmark() -> Result<(), ()> {
    Ok(Ok(()))??;
    Ok(())
}

fn main() {}
