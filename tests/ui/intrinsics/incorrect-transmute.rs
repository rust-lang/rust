fn main() {
    transmute(); // does not ICE
}

extern "rust-intrinsic" fn transmute() {}
//~^ ERROR "rust-intrinsic" ABI is an implementation detail
//~| ERROR intrinsic must be in `extern "rust-intrinsic" { ... }` block
