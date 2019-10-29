// ignore-wasm32-bare compiled with panic=abort by default

// Test that after the call to `std::mem::drop` we do not generate a
// MIR drop of the argument. (We used to have a `DROP(_2)` in the code
// below, as part of bb3.)

fn main() {
    std::mem::drop("".to_string());
}

// END RUST SOURCE
// START rustc.main.ElaborateDrops.before.mir
//    bb2: {
//        StorageDead(_3);
//        _1 = const std::mem::drop::<std::string::String>(move _2) -> [return: bb3, unwind: bb4];
//    }
//    bb3: {
//        StorageDead(_2);
//        StorageDead(_4);
//        StorageDead(_1);
//        _0 = ();
//        return;
//    }
// END rustc.main.ElaborateDrops.before.mir
