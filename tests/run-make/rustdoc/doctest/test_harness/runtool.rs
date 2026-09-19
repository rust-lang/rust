// The whole purpose of this runtool is to pass `--test-threads=1`
// to the inner testsuite to guarantee deterministic output.
// See also #157511.

fn main() {
    let status = std::process::Command::new(std::env::args().nth(1).unwrap())
        .arg("--test-threads=1")
        .status()
        .unwrap();
    std::process::exit(status.code().unwrap())
}
