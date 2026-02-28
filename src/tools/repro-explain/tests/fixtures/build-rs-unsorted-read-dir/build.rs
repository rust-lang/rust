fn main() {
    for entry in std::fs::read_dir("data").unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        println!("cargo::rustc-link-lib={name}");
    }
}
