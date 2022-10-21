// EMIT_MIR issue_101867.main.mir_map.0.mir
fn main() {
    let x: Option<u8> = Some(1);
    let Some(y) = x else {
        panic!();
    };
}
