//@ proc-macro: egui_inspect_derive.rs
//@ revisions: cpass1 cpass2

extern crate egui_inspect_derive;

pub struct TileDef {
    pub layer: (),
    #[cfg(cpass2)]
    pub blend_graphic: String,
}

pub(crate) struct GameState {
    pub(crate) tile_db: TileDb,
}

impl GameState {
    fn inspect_mut(&mut self) {
        egui_inspect_derive::expand! {}
    }
}

fn new() -> GameState {
    loop {}
}

fn main() {
    let mut app = new();
    app.inspect_mut();
}
// this is actually used
pub struct TileDb {
    unknown_bg: TileDef,
}

impl std::fmt::Debug for TileDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        loop {}
    }
}

pub struct PlatformOutput {
    pub copied_text: String,
}

pub fn output_mut<R>(writer: impl FnOnce(&mut PlatformOutput) -> R) -> R {
    loop {}
}
