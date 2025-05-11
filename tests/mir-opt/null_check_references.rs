//@ compile-flags: -C debug-assertions

struct Null {
    a: u32,
}

fn main() {
    // CHECK-LABEL: fn main(
    // CHECK-NOT: {{assert.*}}
    let val: u32 = 42;
    let val_ref: &u32 = &val;
    let _access1: &u32 = &*val_ref;

    let val = Null { a: 42 };
    let _access2: &u32 = &val.a;
}
