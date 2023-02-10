fn main() {
    std::process::Command::new("echo").arg("hello world").spawn().unwrap();
    std::process::Command::new("echo").arg("-n hello").spawn().unwrap();
    std::process::Command::new("cat").arg("--number file").spawn().unwrap();
}
