//! Generates descriptors structure for unstable feature from Unstable Book

use crate::codegen::update;
use crate::codegen::{self, project_root, Mode, Result};
use quote::quote;
use std::fs;
use std::process::Command;
use walkdir::WalkDir;

pub fn generate_unstable_future_descriptor(mode: Mode) -> Result<()> {
    let path = project_root().join(codegen::STORAGE);
    fs::create_dir_all(path.clone())?;

    Command::new("git").current_dir(path.clone()).arg("init").output()?;
    Command::new("git")
        .current_dir(path.clone())
        .args(&["remote", "add", "-f", "origin", codegen::REPOSITORY_URL])
        .output()?;
    Command::new("git")
        .current_dir(path.clone())
        .args(&["sparse-checkout", "set", "/src/doc/unstable-book/src/"])
        .output()?;
    Command::new("git").current_dir(path.clone()).args(&["pull", "origin", "master"]).output()?;
    //FIXME: check git, and do pull

    let src_dir = path.join("src/doc/unstable-book/src");
    let files = WalkDir::new(src_dir.join("language-features"))
        .into_iter()
        .chain(WalkDir::new(src_dir.join("library-features")))
        .filter_map(|e| e.ok())
        .filter(|entry| {
            // Get all `.md ` files
            entry.file_type().is_file()
                && entry.path().extension().map(|ext| ext == "md").unwrap_or(false)
        })
        .collect::<Vec<_>>();

    let definitions = files
        .iter()
        .map(|entry| {
            let path = entry.path();
            let feature_ident =
                format!("{}", path.file_stem().unwrap().to_str().unwrap().replace("-", "_"));
            let doc = format!("{}", std::fs::read_to_string(path).unwrap());

            quote! { LintCompletion { label: #feature_ident, description: #doc } }
        })
        .collect::<Vec<_>>();

    let ts = quote! {
        use crate::completion::LintCompletion;

        #[rustfmt::skip]
        pub const UNSTABLE_FEATURE_DESCRIPTOR:  &[LintCompletion] = &[
            #(#definitions),*
        ];
    };

    let destination = project_root().join(codegen::UNSTABLE_FEATURE);
    let contents = crate::reformat(ts.to_string())?;
    update(destination.as_path(), &contents, mode)?;

    Ok(())
}
