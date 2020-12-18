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
        std::fs::create_dir("target/crater/").unwrap();

        std::fs::create_dir(&krate_download_dir).unwrap();
        std::fs::create_dir(&extract_dir).unwrap();

        let krate_name = format!("{}-{}.crate.tar.gz", &self.name, &self.version);
        let mut krate_dest = std::fs::File::create(krate_download_dir.join(krate_name)).unwrap();
        let mut krate_req = ureq::get(&url).call().unwrap().into_reader();
        std::io::copy(&mut krate_req, &mut krate_dest).unwrap();
        let krate = krate_dest;
        dbg!(&krate);
        let tar = flate2::read::GzDecoder::new(&krate);
        let mut archiv = tar::Archive::new(tar);
        let extracted_path = extract_dir.join(format!("{}-{}", self.name, self.version));
      // println!("ar:  p: {:?}", &krate, extracted_path);
        archiv.unpack(&extracted_path).expect("Failed to extract!");
        // extract

        Krate {
            version: self.version.clone(),
            name: self.name.clone(),
            path: extracted_path,
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
pub fn run() {
    let cargo_clippy_path: PathBuf = PathBuf::from("target/debug/cargo-clippy");
    let clippy_driver_path: PathBuf = PathBuf::from("target/debug/clippy-driver");

    // crates we want to check:
    let krates: Vec<KrateSource> = vec![KrateSource::new("cargo", "0.49.0"), KrateSource::new("regex", "1.4.2")];

    println!("Compiling clippy...");
    build_clippy();
    println!("Done compiling");

    // assert that clippy is found
    assert!(
        cargo_clippy_path.is_file(),
        "target/debug/cargo-clippy binary not found! {}", cargo_clippy_path.display()
    );
    assert!(
        clippy_driver_path.is_file(),
        "target/debug/clippy-driver binary not found! {}", clippy_driver_path.display()
    );

    // download and extract the crates, then run clippy on them and collect clippys warnings
    let _clippy_lint_results: Vec<String> = krates
        .into_iter()
        .map(|krate| krate.download_and_extract())
        .map(|krate| krate.run_clippy_lints())
        .collect::<Vec<String>>();
}
