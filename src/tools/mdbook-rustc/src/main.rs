use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, io};

use mdbook_preprocessor::errors::Error;
use mdbook_rustc::process_main_chapter;
use mdbook_rustc::target::{Target, all_from_rustc_json};

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("supports") => {
            // Supports all renderers.
            return Ok(());
        }
        Some(arg) => {
            return Err(Error::msg(format!("unknown argument: {arg}")));
        }
        None => {}
    }

    let (_ctx, mut book) = mdbook_preprocessor::parse_input(io::stdin().lock())?;
    let rustc_targets = load_rustc_targets()?;
    let target_chapters = book
        .chapters()
        .filter_map(|chapter| {
            Some((
                chapter
                    .source_path
                    .as_ref()
                    .map(|p| p.strip_prefix("platform-support/").ok()?.file_stem()?.to_str())
                    .flatten()?
                    .to_owned(),
                chapter.clone(),
            ))
        })
        .collect::<HashMap<_, _>>();

    book.for_each_chapter_mut(|chapter| {
        if chapter.path.as_deref() == Some("platform-support.md".as_ref()) {
            process_main_chapter(chapter, &rustc_targets, &target_chapters);
        }
    });

    serde_json::to_writer(io::stdout().lock(), &book)?;
    Ok(())
}

fn load_rustc_targets() -> Result<Vec<Target>, Error> {
    let rustc = PathBuf::from(
        env::var("RUSTC").map_err(|_| Error::msg("must pass RUSTC env var pointing to rustc"))?,
    );
    all_from_rustc_json(serde_json::from_str(&rustc_stdout(
        &rustc,
        &["-Z", "unstable-options", "--print", "all-target-specs-json"],
    ))?)
}

fn rustc_stdout(rustc: &Path, args: &[&str]) -> String {
    let output = Command::new(rustc).args(args).env("RUSTC_BOOTSTRAP", "1").output().unwrap();
    if !output.status.success() {
        panic!(
            "rustc failed: {}, {}",
            output.status,
            String::from_utf8(output.stderr).unwrap_or_default()
        )
    }
    String::from_utf8(output.stdout).unwrap()
}
