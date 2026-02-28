fn main() {
    let now = std::time::SystemTime::now();
    let out = std::env::var("OUT_DIR").unwrap();
    std::fs::write(format!("{out}/stamp.txt"), format!("{now:?}")).unwrap();
}
