use super::{LinkerFlavor, Target, wasm32_unknown_emscripten};

pub fn target() -> Result<Target, String> {
    let mut target = wasm32_unknown_emscripten::target()?;
    target.options.post_link_args
        .entry(LinkerFlavor::Em)
        .or_default()
        .extend(vec!["-s".to_string(), "WASM=0".to_string()]);
    Ok(target)
}
