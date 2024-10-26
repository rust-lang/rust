use std::fmt::Write;
use std::fs;

#[path = "../../configure.rs"]
mod configure;
use configure::Config;

fn main() {
    let cfg = Config::from_env();

    list_all_tests(&cfg);

    configure::emit_test_config(&cfg);
}

/// Create a list of all source files in an array. This can be used for making sure that
/// all functions are tested or otherwise covered in some way.
// FIXME: it would probably be better to use rustdoc JSON output to get public functions.
fn list_all_tests(cfg: &Config) {
    let math_src = cfg.manifest_dir.join("../../src/math");

    let mut files = fs::read_dir(math_src)
        .unwrap()
        .map(|f| f.unwrap().path())
        .filter(|entry| entry.is_file())
        .map(|f| f.file_stem().unwrap().to_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    files.sort();

    let mut s = "pub const ALL_FUNCTIONS: &[&str] = &[".to_owned();
    for f in files {
        if f == "mod" {
            // skip mod.rs
            continue;
        }
        write!(s, "\"{f}\",").unwrap();
    }
    write!(s, "];").unwrap();

    let outfile = cfg.out_dir.join("all_files.rs");
    fs::write(outfile, s).unwrap();
}
