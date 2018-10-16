//! A simple tool to generate rust code by passing a ron value
//! to a tera template

#[macro_use]
extern crate failure;
extern crate tera;
extern crate ron;
extern crate heck;

use std::{
    fs,
    collections::HashMap,
    path::Path,
};

use heck::{CamelCase, ShoutySnakeCase, SnakeCase};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Overwrite,
    Verify,
}

pub use Mode::*;

/// A helper to update file on disk if it has changed.
/// With verify = false,
pub fn update(path: &Path, contents: &str, mode: Mode) -> Result<(), failure::Error> {
    match fs::read_to_string(path) {
        Ok(ref old_contents) if old_contents == contents => {
            return Ok(());
        }
        _ => (),
    }
    if mode == Verify {
        bail!("`{}` is not up-to-date", path.display());
    }
    eprintln!("updating {}", path.display());
    fs::write(path, contents)?;
    Ok(())
}

pub fn generate(
    template: &Path,
    src: &Path,
    mode: Mode,
) -> Result<(), failure::Error> {
    assert_eq!(
        template.extension().and_then(|it| it.to_str()), Some("tera"),
        "template file must have .rs.tera extension",
    );
    let file_name = template.file_stem().unwrap().to_str().unwrap();
    assert!(
        file_name.ends_with(".rs"),
        "template file must have .rs.tera extension",
    );
    let tgt = template.with_file_name(file_name);
    let template = fs::read_to_string(template)?;
    let src: ron::Value = {
        let text = fs::read_to_string(src)?;
        ron::de::from_str(&text)?
    };
    let content = render(&template, src)?;
    update(
        &tgt,
        &content,
        mode,
    )
}

pub fn render(
    template: &str,
    src: ron::Value,
) -> Result<String, failure::Error> {
    let mut tera = mk_tera();
    tera.add_raw_template("_src", template)
        .map_err(|e| format_err!("template parsing error: {:?}", e))?;
    let res = tera.render("_src", &src)
        .map_err(|e| format_err!("template rendering error: {:?}", e))?;
    return Ok(res);
}

fn mk_tera() -> tera::Tera {
    let mut res = tera::Tera::default();
    res.register_filter("camel", |arg, _| {
        Ok(arg.as_str().unwrap().to_camel_case().into())
    });
    res.register_filter("snake", |arg, _| {
        Ok(arg.as_str().unwrap().to_snake_case().into())
    });
    res.register_filter("SCREAM", |arg, _| {
        Ok(arg.as_str().unwrap().to_shouty_snake_case().into())
    });
    res.register_function("concat", Box::new(concat));
    res
}

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
