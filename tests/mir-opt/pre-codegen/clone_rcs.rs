//@ compile-flags: -O -C debuginfo=none
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

use std::rc::Rc;

#[repr(align(128))]
pub struct HighAlign([u8; 128]);

// EMIT_MIR clone_rcs.clone_rcs_of_many_types.PreCodegen.after.mir
fn clone_rcs_of_many_types(
    a: &Rc<u8>,
    b: &Rc<u32>,
    c: &Rc<u128>,
    d: &Rc<HighAlign>,
) -> (Rc<u8>, Rc<u32>, Rc<u128>, Rc<HighAlign>) {
    // CHECK-NOT: inlined{{.+}}clone_polymorphic
    // CHECK: clone_polymorphic({{.+}}) ->
    // CHECK: clone_polymorphic({{.+}}) ->
    // CHECK: clone_polymorphic({{.+}}) ->
    // CHECK: clone_polymorphic({{.+}}) ->
    (a.clone(), b.clone(), c.clone(), d.clone())
}
