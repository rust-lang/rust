use std::collections::HashMap;

fn main() {
    let mut libs = HashMap::new();
    libs.insert("zlib", "z");
    libs.insert("ssl", "s");

    for (name, _) in libs.iter() {
        println!("cargo::rustc-link-lib={name}");
    }
}
