use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use clap::{App, Arg, SubCommand};
use failure::bail;

use tools::{
    collect_tests, generate,install_format_hook, run, run_rustfmt,
    Mode, Overwrite, Result, Test, Verify, project_root, run_fuzzer
};

const GRAMMAR_DIR: &str = "crates/ra_syntax/src/grammar";
const OK_INLINE_TESTS_DIR: &str = "crates/ra_syntax/tests/data/parser/inline/ok";
const ERR_INLINE_TESTS_DIR: &str = "crates/ra_syntax/tests/data/parser/inline/err";

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
        .subcommand(SubCommand::with_name("format-hook"))
        .subcommand(SubCommand::with_name("fuzz-tests"))
        .get_matches();
    let mode = if matches.is_present("verify") {
        Verify
    } else {
        Overwrite
    };
    match matches
        .subcommand_name()
        .expect("Subcommand must be specified")
    {
        "install-code" => install_code_extension()?,
        "gen-tests" => gen_tests(mode)?,
        "gen-syntax" => generate(Overwrite)?,
        "format" => run_rustfmt(mode)?,
        "format-hook" => install_format_hook()?,
        "fuzz-tests" => run_fuzzer()?,
        _ => unreachable!(),
    }
    Ok(())
}

fn gen_tests(mode: Mode) -> Result<()> {
    let tests = tests_from_dir(Path::new(GRAMMAR_DIR))?;
    fn install_tests(tests: &HashMap<String, Test>, into: &str, mode: Mode) -> Result<()> {
        let tests_dir = project_root().join(into);
        if !tests_dir.is_dir() {
            fs::create_dir_all(&tests_dir)?;
        }
        // ok is never actually read, but it needs to be specified to create a Test in existing_tests
        let existing = existing_tests(&tests_dir, true)?;
        for t in existing.keys().filter(|&t| !tests.contains_key(t)) {
            panic!("Test is deleted: {}", t);
        }

        let mut new_idx = existing.len() + 1;
        for (name, test) in tests {
            let path = match existing.get(name) {
                Some((path, _test)) => path.clone(),
                None => {
                    let file_name = format!("{:04}_{}.rs", new_idx, name);
                    new_idx += 1;
                    tests_dir.join(file_name)
                }
            };
            teraron::update(&path, &test.text, mode)?;
        }
        Ok(())
    }
    install_tests(&tests.ok, OK_INLINE_TESTS_DIR, mode)?;
    install_tests(&tests.err, ERR_INLINE_TESTS_DIR, mode)
}

#[derive(Default, Debug)]
struct Tests {
    pub ok: HashMap<String, Test>,
    pub err: HashMap<String, Test>,
}

fn tests_from_dir(dir: &Path) -> Result<Tests> {
    let mut res = Tests::default();
    for entry in ::walkdir::WalkDir::new(dir) {
        let entry = entry.unwrap();
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().unwrap_or_default() != "rs" {
            continue;
        }
        process_file(&mut res, entry.path())?;
    }
    let grammar_rs = dir.parent().unwrap().join("grammar.rs");
    process_file(&mut res, &grammar_rs)?;
    return Ok(res);
    fn process_file(res: &mut Tests, path: &Path) -> Result<()> {
        let text = fs::read_to_string(path)?;

        for (_, test) in collect_tests(&text) {
            if test.ok {
                if let Some(old_test) = res.ok.insert(test.name.clone(), test) {
                    bail!("Duplicate test: {}", old_test.name)
                }
            } else {
                if let Some(old_test) = res.err.insert(test.name.clone(), test) {
                    bail!("Duplicate test: {}", old_test.name)
                }
            }
        }
        Ok(())
    }
}

fn existing_tests(dir: &Path, ok: bool) -> Result<HashMap<String, (PathBuf, Test)>> {
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
            ok,
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
        run(
            r"code --install-extension ./ra-lsp-0.0.1.vsix --force",
            "./editors/code",
        )?;
    }
    Ok(())
}
