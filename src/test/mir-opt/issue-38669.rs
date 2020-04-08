// check that we don't StorageDead booleans before they are used

// EMIT_MIR rustc.main.SimplifyCfg-initial.after.mir
fn main() {
    let mut should_break = false;
    loop {
        if should_break {
            break;
        }
        should_break = true;
    }
}
