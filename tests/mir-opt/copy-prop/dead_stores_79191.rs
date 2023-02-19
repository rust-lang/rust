// unit-test: CopyProp

fn id<T>(x: T) -> T {
    x
}

// EMIT_MIR dead_stores_79191.f.CopyProp.after.mir
fn f(mut a: usize) -> usize {
    let b = a;
    a = 5;
    a = b;
    id(a)
}

fn main() {
    f(0);
}
