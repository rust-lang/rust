use super::{wasm32_unknown_emscripten, LinkerFlavor, Target};

pub fn target() -> Target {
    let mut target = wasm32_unknown_emscripten::target();
    target
        .post_link_args
        .entry(LinkerFlavor::Em)
        .or_default()
        .extend(vec!["-s".into(), "WASM=0".into()]);
    target
}
