#![feature(plugin, custom_attribute)]

fn main() {

    #[clippy(author)]
    let x: char = 0x45 as char;
}
