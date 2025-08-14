#![allow(clippy::zombie_processes)]
fn main() {
    // Things it should warn about:
    std::process::Command::new("echo").arg("-n hello").spawn().unwrap();
    //~^ suspicious_command_arg_space

    std::process::Command::new("cat").arg("--number file").spawn().unwrap();
    //~^ suspicious_command_arg_space

    // Things it should not warn about:
    std::process::Command::new("echo").arg("hello world").spawn().unwrap();
    std::process::Command::new("a").arg("--fmt=%a %b %c").spawn().unwrap();
    std::process::Command::new("b").arg("-ldflags=-s -w").spawn().unwrap();
}
