// skip-filecheck
// check that we don't forget to drop the Box if we early return before
// initializing it
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(rustc_attrs, liballoc_internals)]

// EMIT_MIR issue_62289.test.ElaborateDrops.before.mir
fn test() -> Option<Box<u32>> {
    Some(std::boxed::box_new(None?))
}

fn main() {
    test();
}
