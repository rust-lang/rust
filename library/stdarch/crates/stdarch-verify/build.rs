use std::path::Path;

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = dir.parent().unwrap();
    eprintln!("root: {}", root.display());
    walk(&root.join("core_arch/src/x86"));
    walk(&root.join("core_arch/src/x86_64"));
    walk(&root.join("core_arch/src/arm"));
    walk(&root.join("core_arch/src/aarch64"));
}

fn walk(root: &Path) {
    for file in root.read_dir().unwrap() {
        eprintln!("root: {}", root.display());
        let file = file.unwrap();
        if file.file_type().unwrap().is_dir() {
            walk(&file.path());
            continue;
        }
        let path = file.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }

        println!("cargo:rerun-if-changed={}", path.display());
    }
}
