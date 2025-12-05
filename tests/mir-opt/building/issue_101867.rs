//@ compile-flags: -Zmir-opt-level=0
// skip-filecheck
// EMIT_MIR issue_101867.main.built.after.mir
fn main() {
    let x: Option<u8> = Some(1);
    let Some(y) = x else {
        panic!();
    };
}
