// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Z span_free_formats -Z mir-opt-level=4 -C debuginfo=full

// EMIT_MIR inline_trait_method_2.test2.Inline.after.mir
fn test2(x: &dyn X) -> bool {
    // CHECK-LABEL: fn test2(
    // CHECK: (inlined test)
    // CHECK-NOT: (inlined <dyn X as X>::y)
    test(x)
}

#[inline]
fn test(x: &dyn X) -> bool {
    x.y()
}

trait X {
    fn y(&self) -> bool {
        false
    }
}

impl X for () {
    fn y(&self) -> bool {
        true
    }
}

fn main() {
    println!("Should be true: {}", test2(&()));
}
