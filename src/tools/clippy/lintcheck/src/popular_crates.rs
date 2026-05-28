use serde::Deserialize;
use std::error::Error;
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;

#[derive(Deserialize, Debug)]
struct Page {
    crates: Vec<Crate>,
    meta: Meta,
}

#[derive(Deserialize, Debug)]
struct Crate {
    name: String,
    max_version: String,
}

#[derive(Deserialize, Debug)]
struct Meta {
    next_page: String,
}

pub(crate) fn fetch(output: PathBuf, number: usize) -> Result<(), Box<dyn Error>> {
    let agent = ureq::builder()
        .user_agent("clippy/lintcheck (github.com/rust-lang/rust-clippy/)")
        .build();

    let mut crates = Vec::with_capacity(number);
    let mut query = "?sort=recent-downloads&per_page=100".to_string();
    while crates.len() < number {
        let page: Page = agent
            .get(&format!("https://crates.io/api/v1/crates{query}"))
            .call()?
            .into_json()?;

        query = page.meta.next_page;
        crates.extend(page.crates);
        crates.truncate(number);

        let width = number.ilog10() as usize + 1;
        println!("Fetched {:>width$}/{number} crates", crates.len());
    }

    let mut out = "[crates]\n".to_string();
    for Crate { name, max_version } in crates {
        writeln!(out, "{name} = {{ name = '{name}', version = '{max_version}' }}").unwrap();
    }
    fs::write(output, out)?;

    Ok(())
}
