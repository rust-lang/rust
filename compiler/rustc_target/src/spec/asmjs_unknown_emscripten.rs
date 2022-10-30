use super::{wasm32_unknown_emscripten, LinkerFlavor, Target};

pub fn target() -> Target {
    let mut target = wasm32_unknown_emscripten::target();
    target.add_post_link_args(LinkerFlavor::EmCc, &["-sWASM=0", "--memory-init-file", "0"]);
    target
}
