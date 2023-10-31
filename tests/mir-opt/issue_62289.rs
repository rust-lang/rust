// check that we don't forget to drop the Box if we early return before
// initializing it
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(rustc_attrs)]

// EMIT_MIR issue_62289.test.ElaborateDrops.before.mir
fn test() -> Option<Box<u32>> {
    Some(
        #[rustc_box]
        Box::new(None?),
    )
}

fn main() {
    test();
}
