// Extern prelude shadows standard library prelude

#![feature(extern_prelude)]

fn main() {
    let x = Vec::new(0f32, ()); // OK
}
