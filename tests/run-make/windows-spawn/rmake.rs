//@ only-windows

use run_make_support::{run, rustc};

// On Windows `Command` uses `CreateProcessW` to run a new process.
// However, in the past std used to not pass in the application name, leaving
// `CreateProcessW` to use heuristics to guess the intended name from the
// command line string. Sometimes this could go very wrong.
// E.g. in Rust 1.0 `Command::new("foo").arg("bar").spawn()` will try to launch
// `foo bar.exe` if foo.exe does not exist. Which is clearly not desired.

fn main() {
    rustc().input("hello.rs").output("hopefullydoesntexist bar.exe").run();
    rustc().input("spawn.rs").run();
    run("spawn");
}
