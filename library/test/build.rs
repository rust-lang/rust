fn main() {
    println!("cargo:rustc-check-cfg=cfg(enable_unstable_features)");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = std::process::Command::new(rustc).arg("-vV").output().unwrap();
    let stdout = String::from_utf8(version.stdout).unwrap();

    if stdout.contains("nightly") || stdout.contains("dev") {
        println!("cargo:rustc-cfg=enable_unstable_features");
    }
}
