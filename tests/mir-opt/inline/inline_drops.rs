// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
#![crate_type = "lib"]

struct Pair<T, U>(T, U);

// EMIT_MIR inline_drops.half_needs_drop.Inline.diff
pub fn half_needs_drop(s: String) {
    // CHECK-LABEL: fn half_needs_drop(_1: String)
    // CHECK: inlined drop_in_place::<Pair<bool, String>>
    // CHECK: _2 = Pair::<bool, String>({{.+}})
    // CHECK-NOT: drop
    // CHECK: drop((_2.1: std::string::String))
    // CHECK-NOT: drop
    let _x = Pair(true, s);
}
