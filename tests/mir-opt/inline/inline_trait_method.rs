// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// compile-flags: -Z span_free_formats

fn main() {
    println!("{}", test(&()));
}

// EMIT_MIR inline_trait_method.test.Inline.after.mir
fn test(x: &dyn X) -> u32 {
    x.y()
}

trait X {
    fn y(&self) -> u32 {
        1
    }
}

impl X for () {
    fn y(&self) -> u32 {
        2
    }
}
