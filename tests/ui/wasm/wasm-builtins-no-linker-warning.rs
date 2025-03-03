//@ only-wasm32-bare
//@ compile-flags: -Dlinker-messages
//@ build-pass
/// This test checks that linker messages due to gcc lacking
/// support for wasm32 messages are not emitted when the target
/// is wasm32-unknown-unknown or wasm32v1-none

fn main() {
    ()
}
