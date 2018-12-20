use std::path::Path;

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = dir.parent().unwrap();
    walk(&root.join("../coresimd/x86"));
    walk(&root.join("../coresimd/x86_64"));
    walk(&root.join("../coresimd/arm"));
    walk(&root.join("../coresimd/aarch64"));
}

fn walk(root: &Path) {
    for file in root.read_dir().unwrap() {
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
