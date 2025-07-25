//@ compile-flags: -C lto -C panic=abort -Copt-level=3
//@ no-prefer-dynamic

fn main() {
    foo();
}

#[no_mangle]
#[inline(never)]
fn foo() {
    let _a = Box::new(3);
    bar();
    // CHECK-LABEL: define dso_local void @foo
    // CHECK: call void @bar
}

#[inline(never)]
#[no_mangle]
fn bar() {
    println!("hello!");
}
