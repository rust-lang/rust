use std::path::PathBuf;
use std::process::Command;

// represents an archive we download from crates.io
struct KrateSource {
    version: String,
    name: String,
}

// represents the extracted sourcecode of a crate
struct Krate {
    version: String,
    name: String,
}

impl KrateSource {
    fn new(version: &str, name: &str) -> Self {
        KrateSource {
            version: version.into(),
            name: name.into(),
        }
    }
    fn download_and_extract(self) -> Krate {
        // download
        // extract

        Krate {
            version: self.version,
            name: self.name,
        }
    }
}

impl Krate {
    fn run_clippy_lints(&self) -> String {
        todo!();
    }


}

fn build_clippy() {
    Command::new("cargo")
        .arg("build")
        .output()
        .expect("Failed to build clippy!");
}

// the main fn
pub(crate) fn run() {
    let cargo_clippy_path: PathBuf = PathBuf::from("target/debug/cargo-clippy");
    let clippy_driver_path: PathBuf = PathBuf::from("target/debug/cargo-driver");

    // crates we want to check:
    let krates: Vec<KrateSource> = vec![KrateSource::new("cargo", "0.49.0"), KrateSource::new("regex", "1.4.2")];

    build_clippy();
    // assert that clippy is found
    assert!(
        cargo_clippy_path.is_file(),
        "target/debug/cargo-clippy binary not found!"
    );
    assert!(
        clippy_driver_path.is_file(),
        "target/debug/clippy-driver binary not found!"
    );

  let clippy_lint_results: Vec<String> = krates.into_iter().map(|krate|  krate.download_and_extract()).map(|krate| krate.run_clippy_lints()).collect::<Vec<String>>();
}
