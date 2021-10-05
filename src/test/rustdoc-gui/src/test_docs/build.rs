//! generate 2000 constants for testing

use std::{fs::write, path::PathBuf};

fn main() -> std::io::Result<()> {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is not defined");

    let mut output = String::new();
    for i in 0..2000 {
        let line = format!("/// Some const A{0}\npub const A{0}: isize = 0;\n", i);
        output.push_str(&*line);
    };

    write(&[&*out_dir, "huge_amount_of_consts.rs"].iter().collect::<PathBuf>(), output)
}
