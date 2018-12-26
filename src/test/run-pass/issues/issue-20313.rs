// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

#![feature(link_llvm_intrinsics)]

extern {
    #[link_name = "llvm.sqrt.f32"]
    fn sqrt(x: f32) -> f32;
}

fn main(){
}
