use std::fs::remove_dir_all;
use std::path::Path;

use crate::utils::{get_sysroot_dir, remove_file, run_command};

commands_decl! {
    CleanArg => "clean",
    All: "all" => "Clean all data",
    UiTests: "ui-tests" => "Clean ui tests",
}

fn clean_all() -> Result<(), String> {
    let build_sysroot = get_sysroot_dir();
    let dirs_to_remove = [
        "target".into(),
        build_sysroot.join("sysroot"),
        build_sysroot.join("sysroot_src"),
        build_sysroot.join("target"),
    ];
    for dir in dirs_to_remove {
        let _ = remove_dir_all(dir);
    }
    let _ = remove_dir_all(Path::new(crate::BUILD_DIR).join("simple-raytracer"));

    let files_to_remove =
        [build_sysroot.join("Cargo.lock"), "perf.data".into(), "perf.data.old".into()];

    for file in files_to_remove {
        let _ = remove_file(&file);
    }

    println!("Successfully ran `clean all`");
    Ok(())
}

fn clean_ui_tests() -> Result<(), String> {
    let directories = ["run-make", "run-make-cargo", "ui"];
    for directory in directories {
        let path = Path::new(crate::BUILD_DIR)
            .join("rust/build/x86_64-unknown-linux-gnu/test/")
            .join(directory);
        // The directory might not exist, so ignore the error.
        let _ = run_command(&[&"find", &path, &"-name", &"stamp", &"-delete"], None);
    }
    Ok(())
}

pub fn run() -> Result<(), String> {
    // We skip the binary and the "clean" option.
    match CleanArg::from(std::env::args().nth(2).as_deref()) {
        CleanArg::All => clean_all()?,
        CleanArg::UiTests => clean_ui_tests()?,
    }
    Ok(())
}
