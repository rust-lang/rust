use std::path::{Path, PathBuf};

use crate::config::ConfigInfo;
use crate::utils::{git_clone, run_command_with_output};

sub_command_decl! {
    Args => "clone-gcc",
    out_path: PathBuf = "--out-path" => "Location where the GCC repository will be cloned (default: `./gcc`)",
    + config_info: ConfigInfo
    ~ out_path = PathBuf::from("./gcc"),
}

pub fn run() -> Result<(), String> {
    let Some(args) = Args::new()? else {
        return Ok(());
    };

    let result = git_clone("https://github.com/rust-lang/gcc", Some(&args.out_path), false)?;
    if result.ran_clone {
        let gcc_commit = args.config_info.get_gcc_commit()?;
        println!("Checking out GCC commit `{gcc_commit}`...");
        run_command_with_output(
            &[&"git", &"checkout", &gcc_commit],
            Some(Path::new(&result.repo_dir)),
        )?;
    } else {
        println!(
            "There is already a GCC folder in `{}`, leaving things as is...",
            args.out_path.display()
        );
    }
    Ok(())
}
