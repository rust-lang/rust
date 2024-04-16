fn main() {
    read_via_copy();
}

extern "rust-intrinsic" fn read_via_copy() {}
//~^ ERROR intrinsics are subject to change
//~| ERROR intrinsic must be in `extern "rust-intrinsic" { ... }` block
