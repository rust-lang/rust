// Test that the order of fields in struct patterns does not affect the generated MIR.
// When all arms share the same value for a field, we should test that field first,
// regardless of where it appears in the pattern syntax.
// See https://github.com/rust-lang/rust/issues/111563

pub struct P {
    x: u8,
    y: u8,
}

// EMIT_MIR struct_field_order.field_order_x_then_y.SimplifyCfg-initial.after.mir
fn field_order_x_then_y(p: P) -> u32 {
    // Even though `x` is listed first, `y: 0` is shared across all arms, so we should test `y`
    // first to avoid redundant tests.

    // CHECK-LABEL: fn field_order_x_then_y(
    // First test should be on `y` (the shared field).
    // CHECK: switchInt(copy (_1.1: u8)) ->
    // Then test `x` inside the `y == 0` branch.
    // CHECK: switchInt(copy (_1.0: u8)) ->
    match p {
        P { x: 1, y: 0 } => 1,
        P { x: 2, y: 0 } => 2,
        P { x: 3, y: 0 } => 3,
        P { x: 4, y: 0 } => 4,
        _ => 0,
    }
}

// EMIT_MIR struct_field_order.field_order_y_then_x.SimplifyCfg-initial.after.mir
fn field_order_y_then_x(p: P) -> u32 {
    // Here `y` is listed first, so the compiler would have already tested `y` first
    // even without the heuristic. The result should be identical to the above.

    // CHECK-LABEL: fn field_order_y_then_x(
    // CHECK: switchInt(copy (_1.1: u8)) ->
    // CHECK: switchInt(copy (_1.0: u8)) ->
    match p {
        P { y: 0, x: 1 } => 1,
        P { y: 0, x: 2 } => 2,
        P { y: 0, x: 3 } => 3,
        P { y: 0, x: 4 } => 4,
        _ => 0,
    }
}

fn main() {}
