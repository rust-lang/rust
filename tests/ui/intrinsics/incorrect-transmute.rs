fn main() {
    transmute(); // does not ICE
}

extern "rust-intrinsic" fn transmute() {}
//~^ ERROR intrinsics are subject to change
//~| ERROR intrinsic must be in `extern "rust-intrinsic" { ... }` block
