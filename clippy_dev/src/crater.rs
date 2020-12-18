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
    // path to the extracted sources that clippy can check
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
        let krate_download_dir = PathBuf::from("target/crater/downloads");

        // url to download the crate from crates.io
        let url = format!(
            "https://crates.io/api/v1/crates/{}/{}/download",
            self.name, self.version
        );
        println!("Downloading and extracting {} {} from {}", self.name, self.version, url);
        let _ = std::fs::create_dir("target/crater/");
        let _ = std::fs::create_dir(&krate_download_dir);
        let _ = std::fs::create_dir(&extract_dir);

        let krate_file_path = krate_download_dir.join(format!("{}-{}.crate.tar.gz", &self.name, &self.version));
        // don't download/extract if we already have done so
        if !krate_file_path.is_file() {
            // create a file path to download and write the crate data into
            let mut krate_dest = std::fs::File::create(&krate_file_path).unwrap();
            let mut krate_req = ureq::get(&url).call().unwrap().into_reader();
            // copy the crate into the file
            std::io::copy(&mut krate_req, &mut krate_dest).unwrap();

            // unzip the tarball
            let ungz_tar = flate2::read::GzDecoder::new(std::fs::File::open(&krate_file_path).unwrap());
            // extract the tar archive
            let mut archiv = tar::Archive::new(ungz_tar);
            archiv.unpack(&extract_dir).expect("Failed to extract!");

            // unzip the tarball
            let ungz_tar = flate2::read::GzDecoder::new(std::fs::File::open(&krate_file_path).unwrap());
            // extract the tar archive
            let mut archiv = tar::Archive::new(ungz_tar);
            archiv.unpack(&extract_dir).expect("Failed to extract!");
        }
        // crate is extracted, return a new Krate object which contains the path to the extracted
        // sources that clippy can check
        Krate {
            version: self.version.clone(),
            name: self.name.clone(),
            path: extract_dir.join(format!("{}-{}/", self.name, self.version)),
        }
    }
}

impl Krate {
    fn run_clippy_lints(&self, cargo_clippy_path: &PathBuf) -> Vec<String> {
        println!("Linting {} {}...", &self.name, &self.version);
        let cargo_clippy_path = std::fs::canonicalize(cargo_clippy_path).unwrap();

        let all_output = std::process::Command::new(cargo_clippy_path)
            // lint warnings will look like this:
            // src/cargo/ops/cargo_compile.rs:127:35: warning: usage of `FromIterator::from_iter`
            .args(&[
                "--",
                "--message-format=short",
                "--",
                "--cap-lints=warn",
            /*    "--",
                "-Wclippy::pedantic",
                "--",
                "-Wclippy::cargo", */
            ])
            .current_dir(&self.path)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&all_output.stderr);
        let output_lines = stderr.lines();
        let mut output: Vec<String> = output_lines
            .into_iter()
            .filter(|line| line.contains(": warning: "))
            // prefix with the crate name and version
            // cargo-0.49.0/src/cargo/ops/cargo_compile.rs:127:35: warning: usage of `FromIterator::from_iter`
            .map(|line| format!("{}-{}/{}", self.name, self.version, line))
            // remove the "warning: "
            .map(|line| {
                let remove_pat = "warning: ";
                let pos = line
                    .find(&remove_pat)
                    .expect("clippy output did not contain \"warning: \"");
                let mut new = line[0..pos].to_string();
                new.push_str(&line[pos + remove_pat.len()..]);
                new.push('\n');
                new
            })
            .collect();

        // sort messages alphabtically to avoid noise in the logs
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

    // crates we want to check:
    let krates: Vec<KrateSource> = vec![
        // some of these are form cargotest
        KrateSource::new("cargo", "0.49.0"),
        KrateSource::new("iron", "0.6.1"),
        KrateSource::new("ripgrep", "12.1.1"),
        KrateSource::new("tokei", "12.0.4"),
        KrateSource::new("xsv", "0.13.0"),
        KrateSource::new("serde", "1.0.118"),
        KrateSource::new("rayon", "1.5.0"),
        // top 10 crates.io dls
        KrateSource::new("rand", "0.7.3"),
        KrateSource::new("syn", "1.0.54"),
        KrateSource::new("libc", "0.2.81"),
        KrateSource::new("quote", "1.0.7"),
        KrateSource::new("rand_core", "0.6.0"),
        KrateSource::new("unicode-xid", "0.2.1"),
        KrateSource::new("proc-macro2", "1.0.24"),
        KrateSource::new("bitflags", "1.2.1"),
        KrateSource::new("log", "0.4.11"),
        KrateSource::new("regex", "1.4.2")
        //
    ];

    println!("Compiling clippy...");
    build_clippy();
    println!("Done compiling");

    // assert that clippy is found
    assert!(
        cargo_clippy_path.is_file(),
        "target/debug/cargo-clippy binary not found! {}",
        cargo_clippy_path.display()
    );

    // download and extract the crates, then run clippy on them and collect clippys warnings
    let clippy_lint_results: Vec<Vec<String>> = krates
        .into_iter()
        .map(|krate| krate.download_and_extract())
        .map(|krate| krate.run_clippy_lints(&cargo_clippy_path))
        .collect();

    let all_warnings: Vec<String> = clippy_lint_results.into_iter().flatten().collect();

    // save the text into mini-crater/logs.txt
    let text = all_warnings.join("");
    std::fs::write("mini-crater/logs.txt", text).unwrap();
}
