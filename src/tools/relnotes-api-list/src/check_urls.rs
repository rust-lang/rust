//! We generate a lot of URLs in the resulting JSON, and we need all those URLs to be correct. While
//! some URLs are fairly trivial to generate, others are quite tricky (especially `impl` blocks).
//!
//! To ensure we always generate good URLs, we prepare a temporary HTML file containing `<a>` tags for
//! every itme we collected, and we run it through linkchecker. If this fails, it means the code
//! generating URLs has a bug.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Error, anyhow, bail};
use tempfile::tempdir;

use crate::schema::{Schema, SchemaItem};

pub(crate) fn check_urls(schema: &Schema) -> Result<(), Error> {
    let mut urls = Vec::new();
    collect_urls(&mut urls, &schema.items);

    let html_dir = tempdir()?;

    let mut file = File::create(html_dir.path().join("urls.html"))?;
    file.write_all(render_html(&urls).as_bytes())?;

    eprintln!("checking that all generated URLs are valid...");
    let result = Command::new(require_env("LINKCHECKER_PATH")?)
        .arg(html_dir.path())
        .arg("--link-targets-dir")
        .arg(require_env("STD_HTML_DOCS")?)
        .status()?;

    if !result.success() {
        bail!("some URLs are broken, the relnotes-api-list tool has a bug");
    }

    dbg!(require_env("STD_HTML_DOCS")?);

    Ok(())
}

fn collect_urls<'a>(result: &mut Vec<&'a str>, items: &'a [SchemaItem]) {
    for item in items {
        if let Some(url) = &item.url {
            result.push(url);
        }
        collect_urls(result, &item.children);
    }
}

fn render_html(urls: &[&str]) -> String {
    let mut content = "<!DOCTYPE html>\n".to_string();
    for url in urls {
        content.push_str(&format!("<a href=\"{url}\"></a>\n"));
    }
    content
}

fn require_env(name: &str) -> Result<PathBuf, Error> {
    match std::env::var_os(name) {
        Some(value) => Ok(value.into()),
        None => Err(anyhow!("missing environment variable {name}")),
    }
}
