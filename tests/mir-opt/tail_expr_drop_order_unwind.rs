// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// EMIT_MIR tail_expr_drop_order_unwind.method_1.ElaborateDrops.after.mir

#![deny(tail_expr_drop_order)]

use std::backtrace::Backtrace;

#[derive(Clone)]
struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        println!("Drop!");
    }
}

#[derive(Clone)]
struct OtherDrop;
impl Drop for OtherDrop {
    fn drop(&mut self) {
        println!("Drop!");
    }
}

fn method_1(g: Guard) {
    match method_2(&g.clone()) {
        Ok(other_drop) => {
            // repro needs something else being dropped too.
        }
        Err(err) => {}
    }
}

fn method_2(_: &Guard) -> Result<OtherDrop, ()> {
    panic!("Method 2 panics!");
}
