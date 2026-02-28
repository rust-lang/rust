// this tests move up progration, which is not yet implemented

fn foo() -> [u8; 1024] {
    let x = [0; 1024];
    return x;
}

fn main() {}

// File checks to confirm that the array is assigned and returned
// In the mir output

// CHECK: let mut _0: [u8; 1024]; 
// CHECK: _0 = [const 0_u8; 1024];
// CHECK: return;
