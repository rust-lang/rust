fn main() {
    transmute(); // does not ICE
}

extern "rust-intrinsic" fn transmute() {}
//~^ ERROR intrinsic has wrong number of type parameters: found 0, expected 2
//~| ERROR intrinsics are subject to change
//~| ERROR intrinsic must be in `extern "rust-intrinsic" { ... }` block
