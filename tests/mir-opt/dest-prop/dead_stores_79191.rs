// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ unit-test: DestinationPropagation

fn id<T>(x: T) -> T {
    x
}

// EMIT_MIR dead_stores_79191.f.DestinationPropagation.after.mir
fn f(mut a: usize) -> usize {
    // CHECK-LABEL: fn f
    // CHECK: {{_.*}} = {{_.*}}
    let b = a;
    a = 5;
    a = b;
    id(a)
}

fn main() {
    f(0);
}
