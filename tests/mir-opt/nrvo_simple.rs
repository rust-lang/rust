// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: RenameReturnPlace

// EMIT_MIR nrvo_simple.nrvo.RenameReturnPlace.diff
fn nrvo(init: fn(&mut [u8; 1024])) -> [u8; 1024] {
    let mut buf = [0; 1024];
    init(&mut buf);
    buf
}

fn main() {
    let _ = nrvo(|buf| {
        buf[4] = 4;
    });
}
