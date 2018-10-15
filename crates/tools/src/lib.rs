extern crate itertools;
#[macro_use]
extern crate failure;
extern crate ron;
extern crate tera;
extern crate heck;

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};
use itertools::Itertools;
use heck::{CamelCase, ShoutySnakeCase, SnakeCase};

pub type Result<T> = ::std::result::Result<T, failure::Error>;

const GRAMMAR: &str = "ra_syntax/src/grammar.ron";
pub const SYNTAX_KINDS: &str = "ra_syntax/src/syntax_kinds/generated.rs";
pub const SYNTAX_KINDS_TEMPLATE: &str = "ra_syntax/src/syntax_kinds/generated.rs.tera";
pub const AST: &str = "ra_syntax/src/ast/generated.rs";
pub const AST_TEMPLATE: &str = "ra_syntax/src/ast/generated.rs.tera";

#[derive(Debug)]
pub struct Test {
    pub name: String,
    pub text: String,
}

pub fn collect_tests(s: &str) -> Vec<(usize, Test)> {
    let mut res = vec![];
    let prefix = "// ";
    let comment_blocks = s
        .lines()
        .map(str::trim_left)
        .enumerate()
        .group_by(|(_idx, line)| line.starts_with(prefix));

    'outer: for (is_comment, block) in comment_blocks.into_iter() {
        if !is_comment {
            continue;
        }
        let mut block = block.map(|(idx, line)| (idx, &line[prefix.len()..]));

        let (start_line, name) = loop {
            match block.next() {
                Some((idx, line)) if line.starts_with("test ") => {
                    break (idx, line["test ".len()..].to_string())
                }
                Some(_) => (),
                None => continue 'outer,
            }
        };
        let text: String = itertools::join(
            block.map(|(_, line)| line).chain(::std::iter::once("")),
            "\n",
        );
        assert!(!text.trim().is_empty() && text.ends_with("\n"));
        res.push((start_line, Test { name, text }))
    }
    res
}


pub fn update(path: &Path, contents: &str, verify: bool) -> Result<()> {
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

pub fn render_template(template: PathBuf) -> Result<String> {
    let grammar: ron::value::Value = {
        let text = fs::read_to_string(project_root().join(GRAMMAR))?;
        ron::de::from_str(&text)?
    };
    let template = fs::read_to_string(template)?;
    let mut tera = tera::Tera::default();
    tera.add_raw_template("grammar", &template)
        .map_err(|e| format_err!("template error: {:?}", e))?;
    tera.register_function("concat", Box::new(concat));
    tera.register_filter("camel", |arg, _| {
        Ok(arg.as_str().unwrap().to_camel_case().into())
    });
    tera.register_filter("snake", |arg, _| {
        Ok(arg.as_str().unwrap().to_snake_case().into())
    });
    tera.register_filter("SCREAM", |arg, _| {
        Ok(arg.as_str().unwrap().to_shouty_snake_case().into())
    });
    let ret = tera
        .render("grammar", &grammar)
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

pub fn project_root() -> PathBuf {
    Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).parent().unwrap().to_path_buf()
}
