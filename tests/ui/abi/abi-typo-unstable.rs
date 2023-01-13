// rust-intrinsic is unstable and not enabled, so it should not be suggested as a fix
extern "rust-intrinsec" fn rust_intrinsic() {} //~ ERROR invalid ABI

fn main() {
    rust_intrinsic();
}
