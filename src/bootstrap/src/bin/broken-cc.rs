// Show an error message when the wrong C compiler is detected. See the cc_detect module inside of
// bootstrap for more context on why this binary was added.

use std::process::ExitCode;

const PREFIX: &str = "   ";

fn main() -> ExitCode {
    let mut target = None;
    let mut detected_cc = None;
    for arg in std::env::args() {
        match arg.split_once('=') {
            Some(("--broken-cc-target", t)) => target = Some(t.to_string()),
            Some(("--broken-cc-detected", d)) => detected_cc = Some(d.to_string()),
            _ => {}
        }
    }

    let detected_cc = detected_cc.expect("broken-cc not invoked by bootstrap correctly");
    let target = target.expect("broken-cc not invoked by bootstrap correctly");
    let underscore_target = target.replace('-', "_");

    eprintln!();
    eprintln!("{PREFIX}Error: the automatic detection of the C compiler for cross-compiled");
    eprintln!("{PREFIX}target {target} returned the C compiler also used for the");
    eprintln!("{PREFIX}current host platform.");
    eprintln!();
    eprintln!("{PREFIX}This is likely wrong, and will likely result in a broken compilation");
    eprintln!("{PREFIX}artifact. Please specify the correct C compiler for that target, either");
    eprintln!("{PREFIX}with environment variables:");
    eprintln!();
    eprintln!("{PREFIX}    CC_{underscore_target}=path/to/cc");
    eprintln!("{PREFIX}    CXX_{underscore_target}=path/to/cxx");
    eprintln!();
    eprintln!("{PREFIX}...or in config.toml:");
    eprintln!();
    eprintln!("{PREFIX}    [target.\"{target}\"]");
    eprintln!("{PREFIX}    cc = \"path/to/cc\"");
    eprintln!("{PREFIX}    cxx = \"path/to/cxx\"");
    eprintln!();
    eprintln!("{PREFIX}The detected C compiler was:");
    eprintln!();
    eprintln!("{PREFIX}    {detected_cc}");
    eprintln!();

    ExitCode::FAILURE
}
