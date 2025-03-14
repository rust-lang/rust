fn main() {
    read_via_copy();
}

extern "rust-intrinsic" fn read_via_copy() {}
//~^ ERROR "rust-intrinsic" ABI is an implementation detail
//~| ERROR intrinsic must be in `extern "rust-intrinsic" { ... }` block
