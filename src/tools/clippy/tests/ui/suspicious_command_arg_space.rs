fn main() {
    // Things it should warn about:
    std::process::Command::new("echo").arg("-n hello").spawn().unwrap();
    //~^ ERROR: single argument that looks like it should be multiple arguments
    //~| NOTE: `-D clippy::suspicious-command-arg-space` implied by `-D warnings`
    std::process::Command::new("cat").arg("--number file").spawn().unwrap();
    //~^ ERROR: single argument that looks like it should be multiple arguments

    // Things it should not warn about:
    std::process::Command::new("echo").arg("hello world").spawn().unwrap();
    std::process::Command::new("a").arg("--fmt=%a %b %c").spawn().unwrap();
    std::process::Command::new("b").arg("-ldflags=-s -w").spawn().unwrap();
}
