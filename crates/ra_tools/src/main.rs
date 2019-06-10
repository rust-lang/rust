use clap::{App, SubCommand};
use core::str;
use failure::bail;
use ra_tools::{
    generate, gen_tests, install_format_hook, run, run_with_output, run_rustfmt,
    Overwrite, Result, run_fuzzer, run_clippy,
};
use std::{path::{PathBuf}, env};

fn main() -> Result<()> {
    let matches = App::new("tasks")
        .setting(clap::AppSettings::SubcommandRequiredElseHelp)
        .subcommand(SubCommand::with_name("gen-syntax"))
        .subcommand(SubCommand::with_name("gen-tests"))
        .subcommand(SubCommand::with_name("install-code"))
        .subcommand(SubCommand::with_name("format"))
        .subcommand(SubCommand::with_name("format-hook"))
        .subcommand(SubCommand::with_name("fuzz-tests"))
        .subcommand(SubCommand::with_name("lint"))
        .get_matches();
    match matches.subcommand_name().expect("Subcommand must be specified") {
        "install-code" => {
            if cfg!(target_os = "macos") {
                fix_path_for_mac()?;
            }
            install_code_extension()?;
        }
        "gen-tests" => gen_tests(Overwrite)?,
        "gen-syntax" => generate(Overwrite)?,
        "format" => run_rustfmt(Overwrite)?,
        "format-hook" => install_format_hook()?,
        "lint" => run_clippy()?,
        "fuzz-tests" => run_fuzzer()?,
        _ => unreachable!(),
    }
    Ok(())
}

fn install_code_extension() -> Result<()> {
    run("cargo install --path crates/ra_lsp_server --force", ".")?;
    if cfg!(windows) {
        run(r"cmd.exe /c npm.cmd ci", "./editors/code")?;
        run(r"cmd.exe /c npm.cmd run package", "./editors/code")?;
    } else {
        run(r"npm ci", "./editors/code")?;
        run(r"npm run package", "./editors/code")?;
    }
    if cfg!(windows) {
        run(
            r"cmd.exe /c code.cmd --install-extension ./ra-lsp-0.0.1.vsix --force",
            "./editors/code",
        )?;
    } else {
        run(r"code --install-extension ./ra-lsp-0.0.1.vsix --force", "./editors/code")?;
    }
    verify_installed_extensions()?;
    Ok(())
}

fn verify_installed_extensions() -> Result<()> {
    let exts = if cfg!(windows) {
        run_with_output(r"cmd.exe /c code.cmd --list-extensions", ".")?
    } else {
        run_with_output(r"code --list-extensions", ".")?
    };
    if !str::from_utf8(&exts.stdout)?.contains("ra-lsp") {
        bail!(
            "Could not install the Visual Studio Code extension. Please make sure you \
             have at least NodeJS 10.x installed and try again."
        );
    }
    Ok(())
}

fn fix_path_for_mac() -> Result<()> {
    let mut vscode_path: Vec<PathBuf> = {
        const COMMON_APP_PATH: &str =
            r"/Applications/Visual Studio Code.app/Contents/Resources/app/bin";
        const ROOT_DIR: &str = "";
        let home_dir = match env::var("HOME") {
            Ok(home) => home,
            Err(e) => bail!("Failed getting HOME from environment with error: {}.", e),
        };

        [ROOT_DIR, &home_dir]
            .iter()
            .map(|dir| String::from(*dir) + COMMON_APP_PATH)
            .map(PathBuf::from)
            .filter(|path| path.exists())
            .collect()
    };

    if !vscode_path.is_empty() {
        let vars = match env::var_os("PATH") {
            Some(path) => path,
            None => bail!("Could not get PATH variable from env."),
        };

        let mut paths = env::split_paths(&vars).collect::<Vec<_>>();
        paths.append(&mut vscode_path);
        let new_paths = env::join_paths(paths)?;
        env::set_var("PATH", &new_paths);
    }

    Ok(())
}
