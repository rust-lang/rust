// compile-flags: -Z identify_regions

// Tests to make sure we correctly generate falseUnwind edges in loops

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

// END RUST SOURCE
// START rustc.main.SimplifyCfg-qualify-consts.after.mir
//    ...
//    bb1 (cleanup): {
//        resume;
//    }
//    ...
//    bb3: { // Entry into the loop
//        _1 = ();
//        StorageDead(_2);
//        StorageDead(_1);
//        goto -> bb5;
//    }
//    ...
//    bb5: { // The loop_block
//        falseUnwind -> [real: bb6, cleanup: bb1];
//    }
//    bb6: { // The loop body (body_block)
//        StorageLive(_6);
//        _6 = const 1i32;
//        FakeRead(ForLet, _6);
//        StorageDead(_6);
//        goto -> bb5;
//    }
//    ...
// END rustc.main.SimplifyCfg-qualify-consts.after.mir
