use super::wasm_base;
use super::Target;


pub fn target() -> Target {
    Target {
        llvm_target: "wasm64-unknown-emscripten".into(),
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-p10:8:8-p20:8:8-i64:64-f128:64-n32:64-S128-ni:1:10:20".into(),
        arch: "wasm64".into(),
        options: wasm_base::options(),
    }
}
