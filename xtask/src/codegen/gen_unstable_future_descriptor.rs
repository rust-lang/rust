//! Generates descriptors structure for unstable feature from Unstable Book

use crate::codegen::{update, reformat};
use crate::codegen::{self, project_root, Mode, Result};
use crate::not_bash::{fs2, pushd, run};
use proc_macro2::TokenStream;
use quote::quote;
use std::path::PathBuf;
use walkdir::WalkDir;

fn generate_descriptor(src_dir: PathBuf) -> Result<TokenStream> {
    let files = WalkDir::new(src_dir.join("language-features"))
        .into_iter()
        .chain(WalkDir::new(src_dir.join("library-features")))
        .filter_map(|e| e.ok())
        .filter(|entry| {
            // Get all `.md ` files
            entry.file_type().is_file() && entry.path().extension().unwrap_or_default() == "md"
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

        pub(crate) const UNSTABLE_FEATURE_DESCRIPTOR:  &[LintCompletion] = &[
            #(#definitions),*
        ];
    };
    Ok(ts)
}

pub fn generate_unstable_future_descriptor(mode: Mode) -> Result<()> {
    let path = project_root().join(codegen::STORAGE);
    fs2::create_dir_all(path.clone())?;

    let _d = pushd(path.clone());
    run!("git init")?;
    run!("git remote add -f origin {}", codegen::REPOSITORY_URL)?;
    run!("git pull origin master")?;

    let src_dir = path.join(codegen::REPO_PATH);
    let content = generate_descriptor(src_dir)?.to_string();

    let contents = reformat(content)?;
    let destination = project_root().join(codegen::UNSTABLE_FEATURE);
    update(destination.as_path(), &contents, mode)?;

    Ok(())
}
