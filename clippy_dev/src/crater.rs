use std::path::PathBuf;
use std::process::Command;

// represents an archive we download from crates.io
#[derive(Debug)]
struct KrateSource {
    version: String,
    name: String,
}

// represents the extracted sourcecode of a crate
#[derive(Debug)]
struct Krate {
    version: String,
    name: String,
    path: PathBuf,
}

impl KrateSource {
    fn new(name: &str, version: &str) -> Self {
        KrateSource {
            version: version.into(),
            name: name.into(),
        }
    }
    fn download_and_extract(&self) -> Krate {
        let extract_dir = PathBuf::from("target/crater/crates");

        // download
        let krate_download_dir = PathBuf::from("target/crater/downloads");

        let url = format!(
            "https://crates.io/api/v1/crates/{}/{}/download",
            self.name, self.version
        );
        println!("Downloading {}, {} / {}", self.name, self.version, url);
        let _ = std::fs::create_dir("target/crater/");

        let _ = std::fs::create_dir(&krate_download_dir);
        let _ = std::fs::create_dir(&extract_dir);

        let krate_name = format!("{}-{}.crate.tar.gz", &self.name, &self.version);
        let krate_file_path = krate_download_dir.join(krate_name);
        let mut krate_dest = std::fs::File::create(&krate_file_path).unwrap();
        let mut krate_req = ureq::get(&url).call().unwrap().into_reader();
        std::io::copy(&mut krate_req, &mut krate_dest).unwrap();
        // unzip the tarball
        let dl = std::fs::File::open(krate_file_path).unwrap();

        let ungz_tar = flate2::read::GzDecoder::new(dl);
        // extract the tar archive
        let mut archiv = tar::Archive::new(ungz_tar);
        let extract_path = extract_dir.clone();
        archiv.unpack(&extract_path).expect("Failed to extract!");
        // extracted
        dbg!(&extract_path);
        Krate {
            version: self.version.clone(),
            name: self.name.clone(),
            path: extract_dir.join(format!("{}-{}/", self.name, self.version)),
        }
    }
}

impl Krate {
    fn run_clippy_lints(&self, cargo_clippy_path: &PathBuf) -> Vec<String> {
        let cargo_clippy_path = std::fs::canonicalize(cargo_clippy_path).unwrap();
        let project_root = &self.path;
        dbg!(&cargo_clippy_path);
        dbg!(&project_root);

        let output = std::process::Command::new(cargo_clippy_path)
            .args(&["--", "--message-format=short", "--", "--cap-lints=warn"])
            .current_dir(project_root)
            .output()
            .unwrap();
        let mut output = String::from_utf8_lossy(&output.stderr);
        let output: Vec<&str> = output.lines().collect();
        let mut output: Vec<String> = output
            .into_iter()
            .filter(|line| line.contains(": warning: "))
            .map(|l| l.to_string())
            .collect();

        output.sort();
        output
    }
}

fn build_clippy() {
    Command::new("cargo")
        .arg("build")
        .output()
        .expect("Failed to build clippy!");
}

// the main fn
pub fn run() {
    let cargo_clippy_path: PathBuf = PathBuf::from("target/debug/cargo-clippy");
    let clippy_driver_path: PathBuf = PathBuf::from("target/debug/clippy-driver");

    // crates we want to check:
    let krates: Vec<KrateSource> = vec![KrateSource::new("regex", "1.4.2"), KrateSource::new("cargo", "0.49.0")];

    println!("Compiling clippy...");
    build_clippy();
    println!("Done compiling");

    // assert that clippy is found
    assert!(
        cargo_clippy_path.is_file(),
        "target/debug/cargo-clippy binary not found! {}",
        cargo_clippy_path.display()
    );
    assert!(
        clippy_driver_path.is_file(),
        "target/debug/clippy-driver binary not found! {}",
        clippy_driver_path.display()
    );

    // download and extract the crates, then run clippy on them and collect clippys warnings
    let clippy_lint_results: Vec<Vec<String>> = krates
        .into_iter()
        .map(|krate| krate.download_and_extract())
        .map(|krate| krate.run_clippy_lints(&cargo_clippy_path))
        .collect();

    let results: Vec<String> = clippy_lint_results.into_iter().flatten().collect();

    results.iter().for_each(|l| println!("{}", l));
}
