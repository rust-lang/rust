extern crate clap;
extern crate failure;
extern crate tools;
extern crate walkdir;
extern crate teraron;

use clap::{App, Arg, SubCommand};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};
use tools::{
    collect_tests, Result, Test, generate, Mode, Overwrite, Verify, run, run_rustfmt,
};
use failure::bail;

const GRAMMAR_DIR: &str = "./crates/ra_syntax/src/grammar";
const INLINE_TESTS_DIR: &str = "./crates/ra_syntax/tests/data/parser/inline";

fn main() -> Result<()> {
    let matches = App::new("tasks")
        .setting(clap::AppSettings::SubcommandRequiredElseHelp)
        .arg(
            Arg::with_name("verify")
                .long("--verify")
                .help("Verify that generated code is up-to-date")
                .global(true),
        )
        .subcommand(SubCommand::with_name("gen-syntax"))
        .subcommand(SubCommand::with_name("gen-tests"))
        .subcommand(SubCommand::with_name("install-code"))
        .subcommand(SubCommand::with_name("format"))
        .get_matches();
    let mode = if matches.is_present("verify") {
        Verify
    } else {
        Overwrite
    };
    match matches.subcommand() {
        ("install-code", _) => install_code_extension()?,
        ("gen-tests", _) => gen_tests(mode)?,
        ("gen-syntax", _) => generate(Overwrite)?,
        ("format", _) => run_rustfmt(Overwrite)?,
        _ => unreachable!(),
    }
    Ok(())
}

fn gen_tests(mode: Mode) -> Result<()> {
    let tests = tests_from_dir(Path::new(GRAMMAR_DIR))?;

    let inline_tests_dir = Path::new(INLINE_TESTS_DIR);
    if !inline_tests_dir.is_dir() {
        fs::create_dir_all(inline_tests_dir)?;
    }
    let existing = existing_tests(inline_tests_dir)?;

    for t in existing.keys().filter(|&t| !tests.contains_key(t)) {
        panic!("Test is deleted: {}", t);
    }

    let mut new_idx = existing.len() + 2;
    for (name, test) in tests {
        let path = match existing.get(&name) {
            Some((path, _test)) => path.clone(),
            None => {
                let file_name = format!("{:04}_{}.rs", new_idx, name);
                new_idx += 1;
                inline_tests_dir.join(file_name)
            }
        };
        teraron::update(&path, &test.text, mode)?;
    }
    Ok(())
}

fn tests_from_dir(dir: &Path) -> Result<HashMap<String, Test>> {
    let mut res = HashMap::new();
    for entry in ::walkdir::WalkDir::new(dir) {
        let entry = entry.unwrap();
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().unwrap_or_default() != "rs" {
            continue;
        }
        let text = fs::read_to_string(entry.path())?;

        for (_, test) in collect_tests(&text) {
            if let Some(old_test) = res.insert(test.name.clone(), test) {
                bail!("Duplicate test: {}", old_test.name)
            }
        }
    }
    Ok(res)
}

fn existing_tests(dir: &Path) -> Result<HashMap<String, (PathBuf, Test)>> {
    let mut res = HashMap::new();
    for file in fs::read_dir(dir)? {
        let file = file?;
        let path = file.path();
        if path.extension().unwrap_or_default() != "rs" {
            continue;
        }
        let name = {
            let file_name = path.file_name().unwrap().to_str().unwrap();
            file_name[5..file_name.len() - 3].to_string()
        };
        let text = fs::read_to_string(&path)?;
        let test = Test {
            name: name.clone(),
            text,
        };
        if let Some(old) = res.insert(name, (path, test)) {
            println!("Duplicate test: {:?}", old);
        }
    }
    Ok(res)
}

fn install_code_extension() -> Result<()> {
    run("cargo install --path crates/ra_lsp_server --force", ".")?;
    if cfg!(windows) {
        run(r"cmd.exe /c npm.cmd install", "./editors/code")?;
    } else {
        run(r"npm install", "./editors/code")?;
    }
    run(
        r"node ./node_modules/vsce/out/vsce package",
        "./editors/code",
    )?;
    if cfg!(windows) {
        run(
            r"cmd.exe /c code.cmd --install-extension ./ra-lsp-0.0.1.vsix",
            "./editors/code",
        )?;
    } else {
        run(
            r"code --install-extension ./ra-lsp-0.0.1.vsix",
            "./editors/code",
        )?;
    }
    Ok(())
}
