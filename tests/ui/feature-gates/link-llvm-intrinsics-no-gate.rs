//! <https://github.com/rust-lang/rust/issues/20313>.

extern "C" {
    #[link_name = "llvm.sqrt.f32"]
    fn sqrt(x: f32) -> f32; //~ ERROR linking to LLVM intrinsics is experimental
}

fn main() {}
