// skip-filecheck
// check that we don't forget to drop the Box if we early return before
// initializing it
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(rustc_attrs, liballoc_internals)]

// EMIT_MIR issue_62289.test.ElaborateDrops.after.mir
fn test() -> Option<Vec<u32>> {
    Some(vec![None?])
}

fn main() {
    test();
}
