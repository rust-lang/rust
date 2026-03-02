//@ test-mir-pass: DestinationPropagation

// File checks to confirm that the array is assigned and returned
// In the mir output

fn foo() -> [u8; 1024] {
    let x = [0; 1024];
    return x;
}

// CHECK-LABEL: fn foo(
// CHECK: let mut _0: [u8; 1024];
// CHECK: _0 = [const 0_u8; 1024];
// CHECK: return;

fn main() {}
