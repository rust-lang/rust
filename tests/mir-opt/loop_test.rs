// skip-filecheck
//@ compile-flags: -Z identify_regions

// Tests to make sure we correctly generate falseUnwind edges in loops

// EMIT_MIR loop_test.main.SimplifyCfg-promote-consts.after.mir
fn main() {
    // Exit early at runtime. Since only care about the generated MIR
    // and not the runtime behavior (which is exercised by other tests)
    // we just bail early. Without this the test just loops infinitely.
    if true {
        return;
    }
    loop {
        let x = 1;
        continue;
    }
}
