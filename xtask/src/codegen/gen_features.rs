//! Generates descriptors structure for unstable feature from Unstable Book
use std::path::{Path, PathBuf};

use quote::quote;
use walkdir::WalkDir;
use xshell::{cmd, read_file};

use crate::codegen::{project_root, reformat, update, Mode, Result};

pub fn generate_features(mode: Mode) -> Result<()> {
    if !Path::new("./target/rust").exists() {
        cmd!("git clone https://github.com/rust-lang/rust ./target/rust").run()?;
    }

    let contents = generate_descriptor("./target/rust/src/doc/unstable-book/src".into())?;

    let destination = project_root().join("crates/ide/src/completion/generated_features.rs");
    update(destination.as_path(), &contents, mode)?;

    Ok(())
}

fn generate_descriptor(src_dir: PathBuf) -> Result<String> {
    let definitions = ["language-features", "library-features"]
        .iter()
        .flat_map(|it| WalkDir::new(src_dir.join(it)))
        .filter_map(|e| e.ok())
        .filter(|entry| {
            // Get all `.md ` files
            entry.file_type().is_file() && entry.path().extension().unwrap_or_default() == "md"
        })
        .map(|entry| {
            let path = entry.path();
            let feature_ident = path.file_stem().unwrap().to_str().unwrap().replace("-", "_");
            let doc = read_file(path).unwrap();

            quote! { LintCompletion { label: #feature_ident, description: #doc } }
        });

    let ts = quote! {
        use crate::completion::complete_attribute::LintCompletion;

        pub(super) const FEATURES:  &[LintCompletion] = &[
            #(#definitions),*
        ];
    };
    reformat(&ts.to_string())
}
