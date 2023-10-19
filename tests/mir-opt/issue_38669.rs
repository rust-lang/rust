// skip-filecheck
// check that we don't StorageDead booleans before they are used

// EMIT_MIR issue_38669.main.SimplifyCfg-initial.after.mir
fn main() {
    let mut should_break = false;
    loop {
        if should_break {
            break;
        }
        should_break = true;
    }
}
