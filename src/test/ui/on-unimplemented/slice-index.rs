// Test new Index error message for slices
// ignore-tidy-linelength



use std::ops::Index;


fn main() {
    let x = &[1, 2, 3] as &[i32];
    x[1i32]; //~ ERROR E0277
    x[..1i32]; //~ ERROR E0277
}
