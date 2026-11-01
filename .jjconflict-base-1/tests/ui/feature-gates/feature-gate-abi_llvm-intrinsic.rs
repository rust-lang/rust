extern "llvm-intrinsic" fn foo() {
    //~^ ERROR: "llvm-intrinsic" ABI is an implementation detail and perma-unstable
}

fn main() {
    foo();
}
