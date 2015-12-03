extern crate compiletest_rs as compiletest;

use std::path::Path;
use std::env::var;

#[test]
fn dogfood() {
    let mut config = compiletest::default_config();

    let cfg_mode = "run-pass".parse().ok().expect("Invalid mode");
    let mut s = String::new();
    s.push_str(" -L target/debug/");
    s.push_str(" -L target/debug/deps");
    s.push_str(" -Zextra-plugins=clippy -Ltarget_recur/debug -Dclippy_pedantic -Dclippy");
    config.target_rustcflags = Some(s);
    if let Ok(name) = var::<&str>("TESTNAME") {
        let s : String = name.to_owned();
        config.filter = Some(s)
    }

    config.mode = cfg_mode;

    compiletest::runtest::run(config, &Path::new("src/lib.rs"));
}
