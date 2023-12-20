use crate::utils::remove_file;

use std::fs::remove_dir_all;

#[derive(Default)]
struct CleanArg {
    all: bool,
}

impl CleanArg {
    fn new() -> Result<Option<Self>, String> {
        let mut args = CleanArg::default();

        // We skip the binary and the "clean" option.
        for arg in std::env::args().skip(2) {
            match arg.as_str() {
                "all" => args.all = true,
                "--help" => {
                    Self::usage();
                    return Ok(None);
                }
                a => return Err(format!("Unknown argument `{}`", a)),
            }
        }
        Ok(Some(args))
    }

    fn usage() {
        println!(
            r#"
    `clean` command help:

        all                      : Clean all data
        --help                   : Show this help
    "#
        )
    }
}

fn clean_all() -> Result<(), String> {
    let dirs_to_remove = [
        "target",
        "build_sysroot/sysroot",
        "build_sysroot/sysroot_src",
        "build_sysroot/target",
        "regex",
        "simple-raytracer",
    ];
    for dir in dirs_to_remove {
        let _ = remove_dir_all(dir);
    }

    let files_to_remove = ["build_sysroot/Cargo.lock", "perf.data", "perf.data.old"];

    for file in files_to_remove {
        let _ = remove_file(file);
    }

    println!("Successfully ran `clean all`");
    Ok(())
}

pub fn run() -> Result<(), String> {
    let args = match CleanArg::new()? {
        Some(a) => a,
        None => return Ok(()),
    };

    if args.all {
        clean_all()?;
    }
    Ok(())
}
