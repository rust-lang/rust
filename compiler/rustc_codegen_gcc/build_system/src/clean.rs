use crate::utils::{remove_file, run_command};

use std::fs::remove_dir_all;
use std::path::Path;

#[derive(Default)]
enum CleanArg {
    /// `clean all`
    All,
    /// `clean ui-tests`
    UiTests,
    /// `clean --help`
    #[default]
    Help,
}

impl CleanArg {
    fn new() -> Result<Self, String> {
        // We skip the binary and the "clean" option.
        for arg in std::env::args().skip(2) {
            return match arg.as_str() {
                "all" => Ok(Self::All),
                "ui-tests" => Ok(Self::UiTests),
                "--help" => Ok(Self::Help),
                a => Err(format!("Unknown argument `{}`", a)),
            };
        }
        Ok(Self::default())
    }
}

fn usage() {
    println!(
        r#"
`clean` command help:

    all                      : Clean all data
    ui-tests                 : Clean ui tests
    --help                   : Show this help
"#
    )
}

fn clean_all() -> Result<(), String> {
    let dirs_to_remove = [
        "target",
        "build_sysroot/sysroot",
        "build_sysroot/sysroot_src",
        "build_sysroot/target",
    ];
    for dir in dirs_to_remove {
        let _ = remove_dir_all(dir);
    }
    let dirs_to_remove = ["regex", "rand", "simple-raytracer"];
    for dir in dirs_to_remove {
        let _ = remove_dir_all(Path::new(crate::BUILD_DIR).join(dir));
    }

    let files_to_remove = ["build_sysroot/Cargo.lock", "perf.data", "perf.data.old"];

    for file in files_to_remove {
        let _ = remove_file(file);
    }

    println!("Successfully ran `clean all`");
    Ok(())
}

fn clean_ui_tests() -> Result<(), String> {
    let path = Path::new(crate::BUILD_DIR).join("rust/build/x86_64-unknown-linux-gnu/test/ui/");
    run_command(&[&"find", &path, &"-name", &"stamp", &"-delete"], None)?;
    Ok(())
}

pub fn run() -> Result<(), String> {
    match CleanArg::new()? {
        CleanArg::All => clean_all()?,
        CleanArg::UiTests => clean_ui_tests()?,
        CleanArg::Help => usage(),
    }
    Ok(())
}
