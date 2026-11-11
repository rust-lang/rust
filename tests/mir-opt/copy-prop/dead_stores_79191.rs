// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: CopyProp

fn id<T>(x: T) -> T {
    x
}

// EMIT_MIR dead_stores_79191.f.CopyProp.after.mir
fn f(mut a: usize) -> usize {
    // CHECK-LABEL: fn f(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];
    // CHECK: [[b]] = copy [[a]];
    // CHECK: [[a]] = const 5_usize;
    // CHECK: [[a]] = copy [[b]];
    // CHECK: [[c:_.*]] = copy [[a]]
    // CHECK: id::<usize>(move [[c]])
    let b = a;
    a = 5;
    a = b;
    id(a)
}

fn main() {
    f(0);
}
