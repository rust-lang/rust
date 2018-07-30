extern crate clap;
#[macro_use]
extern crate failure;
extern crate ron;
extern crate tera;
extern crate walkdir;
extern crate tools;
#[macro_use]
extern crate commandspec;

use std::{collections::{HashSet, HashMap}, fs, path::Path};
use clap::{App, Arg, SubCommand};
use tools::{collect_tests, Test};

type Result<T> = ::std::result::Result<T, failure::Error>;

const GRAMMAR_DIR: &str = "./src/parser/grammar";
const INLINE_TESTS_DIR: &str = "tests/data/parser/inline";
const GRAMMAR: &str = "./src/grammar.ron";
const SYNTAX_KINDS: &str = "./src/syntax_kinds/generated.rs";
const SYNTAX_KINDS_TEMPLATE: &str = "./src/syntax_kinds/generated.rs.tera";

fn main() -> Result<()> {
    let matches = App::new("tasks")
        .setting(clap::AppSettings::SubcommandRequiredElseHelp)
        .arg(
            Arg::with_name("verify")
                .long("--verify")
                .help("Verify that generated code is up-to-date")
                .global(true),
        )
        .subcommand(SubCommand::with_name("gen-kinds"))
        .subcommand(SubCommand::with_name("gen-tests"))
        .subcommand(SubCommand::with_name("install-code"))
        .get_matches();
    match matches.subcommand() {
        ("install-code", _) => install_code_extension()?,
        (name, Some(matches)) => run_gen_command(name, matches.is_present("verify"))?,
        _ => unreachable!(),
    }
    Ok(())
}

fn run_gen_command(name: &str, verify: bool) -> Result<()> {
    match name {
        "gen-kinds" => update(Path::new(SYNTAX_KINDS), &get_kinds()?, verify),
        "gen-tests" => gen_tests(verify),
        _ => unreachable!(),
    }
}

fn update(path: &Path, contents: &str, verify: bool) -> Result<()> {
    match fs::read_to_string(path) {
        Ok(ref old_contents) if old_contents == contents => {
            return Ok(());
        }
        _ => (),
    }
    if verify {
        bail!("`{}` is not up-to-date", path.display());
    }
    eprintln!("updating {}", path.display());
    fs::write(path, contents)?;
    Ok(())
}

fn get_kinds() -> Result<String> {
    let grammar = grammar()?;
    let template = fs::read_to_string(SYNTAX_KINDS_TEMPLATE)?;
    let mut tera = tera::Tera::default();
    tera.add_raw_template("grammar", &template)
        .map_err(|e| format_err!("template error: {:?}", e))?;
    tera.register_global_function("concat", Box::new(concat));
    let ret = tera.render("grammar", &grammar)
        .map_err(|e| format_err!("template error: {:?}", e))?;
    return Ok(ret);

    fn concat(args: HashMap<String, tera::Value>) -> tera::Result<tera::Value> {
        let mut elements = Vec::new();
        for &key in ["a", "b", "c"].iter() {
            let val = match args.get(key) {
                Some(val) => val,
                None => continue,
            };
            let val = val.as_array().unwrap();
            elements.extend(val.iter().cloned());
        }
        Ok(tera::Value::Array(elements))
    }
}

fn grammar() -> Result<ron::value::Value> {
    let text = fs::read_to_string(GRAMMAR)?;
    let ret = ron::de::from_str(&text)?;
    Ok(ret)
}

fn gen_tests(verify: bool) -> Result<()> {
    let tests = tests_from_dir(Path::new(GRAMMAR_DIR))?;

    let inline_tests_dir = Path::new(INLINE_TESTS_DIR);
    if !inline_tests_dir.is_dir() {
        fs::create_dir_all(inline_tests_dir)?;
    }
    let existing = existing_tests(inline_tests_dir)?;

    for t in existing.difference(&tests) {
        panic!("Test is deleted: {}\n{}", t.name, t.text);
    }

    let new_tests = tests.difference(&existing);
    for (i, t) in new_tests.enumerate() {
        let name = format!("{:04}_{}.rs", existing.len() + i + 1, t.name);
        let path = inline_tests_dir.join(name);
        update(&path, &t.text, verify)?;
    }
    Ok(())
}

fn tests_from_dir(dir: &Path) -> Result<HashSet<Test>> {
    let mut res = HashSet::new();
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
            if let Some(old_test) = res.replace(test) {
                bail!("Duplicate test: {}", old_test.name)
            }
        }
    }
    Ok(res)
}

fn existing_tests(dir: &Path) -> Result<HashSet<Test>> {
    let mut res = HashSet::new();
    for file in fs::read_dir(dir)? {
        let file = file?;
        let path = file.path();
        if path.extension().unwrap_or_default() != "rs" {
            continue;
        }
        let name = path.file_name().unwrap().to_str().unwrap();
        let name = name["0000_".len()..name.len() - 3].to_string();
        let text = fs::read_to_string(&path)?;
        res.insert(Test { name, text });
    }
    Ok(res)
}

fn install_code_extension() -> Result<()> {
    execute!(r"
cd code
npm install
    ")?;
    execute!(r"
cd code
./node_modules/vsce/out/vsce package
    ")?;
    execute!(r"
cd code
code --install-extension ./libsyntax-rust-0.0.1.vsix
    ")?;
    Ok(())
}
