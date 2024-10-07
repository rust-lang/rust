//@ compile-flags: -O

#![crate_type = "lib"]

#[no_mangle]
// CHECK-LABEL: @noop(
pub fn noop(v: &mut Vec<u8>) {
    // CHECK-NOT: grow_one
    // CHECK-NOT: call
    // CHECK: tail call void @llvm.assume
    // CHECK-NOT: grow_one
    // CHECK-NOT: call {{(v(o(i(d( (@(l(l(v(m(\.(a(s(s(u(m([^e]|$)|[^m]|$)|[^u]|$)|[^s]|$)|[^s]|$)|[^a]|$)|[^.]|$)|[^m]|$)|[^v]|$)|[^l]|$)|[^l]|$)|[^@]|$)|[^ ]|$)|[^d]|$)|[^i]|$)|[^o]|$)|[^v]|$)}}
    // CHECK: ret
    if let Some(x) = v.pop() {
        v.push(x)
    }
}

#[no_mangle]
// CHECK-LABEL: @push_byte(
pub fn push_byte(v: &mut Vec<u8>) {
    // CHECK: call {{.*}}grow_one
    v.push(3);
}
