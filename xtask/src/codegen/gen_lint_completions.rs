//! Generates descriptors structure for unstable feature from Unstable Book
use std::path::{Path, PathBuf};

use quote::quote;
use walkdir::WalkDir;
use xshell::{cmd, read_file};

use crate::{
    codegen::{project_root, reformat, update, Mode, Result},
    run_rustfmt,
};

pub fn generate_lint_completions(mode: Mode) -> Result<()> {
    if !Path::new("./target/rust").exists() {
        cmd!("git clone --depth=1 https://github.com/rust-lang/rust ./target/rust").run()?;
    }

    let ts_features = generate_descriptor("./target/rust/src/doc/unstable-book/src".into())?;
    cmd!("curl http://rust-lang.github.io/rust-clippy/master/lints.json --output ./target/clippy_lints.json").run()?;

    let ts_clippy = generate_descriptor_clippy(&Path::new("./target/clippy_lints.json"))?;
    let ts = quote! {
        use crate::completions::attribute::LintCompletion;
        #ts_features
        #ts_clippy
    };
    let contents = reformat(ts.to_string().as_str())?;

    let destination = project_root().join("crates/completion/src/generated_lint_completions.rs");
    update(destination.as_path(), &contents, mode)?;
    run_rustfmt(mode)?;

    Ok(())
}

fn generate_descriptor(src_dir: PathBuf) -> Result<proc_macro2::TokenStream> {
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
        pub(super) const FEATURES:  &[LintCompletion] = &[
            #(#definitions),*
        ];
    };

    Ok(ts)
}

#[derive(Default)]
struct ClippyLint {
    help: String,
    id: String,
}

fn generate_descriptor_clippy(path: &Path) -> Result<proc_macro2::TokenStream> {
    let file_content = read_file(path)?;
    let mut clippy_lints: Vec<ClippyLint> = vec![];

    for line in file_content.lines().map(|line| line.trim()) {
        if line.starts_with(r#""id":"#) {
            let clippy_lint = ClippyLint {
                id: line
                    .strip_prefix(r#""id": ""#)
                    .expect("should be prefixed by id")
                    .strip_suffix(r#"","#)
                    .expect("should be suffixed by comma")
                    .into(),
                help: String::new(),
            };
            clippy_lints.push(clippy_lint)
        } else if line.starts_with(r#""What it does":"#) {
            // Typical line to strip: "What is doest": "Here is my useful content",
            let prefix_to_strip = r#""What it does": ""#;
            let suffix_to_strip = r#"","#;

            let clippy_lint = clippy_lints.last_mut().expect("clippy lint must already exist");
            clippy_lint.help = line
                .strip_prefix(prefix_to_strip)
                .expect("should be prefixed by what it does")
                .strip_suffix(suffix_to_strip)
                .expect("should be suffixed by comma")
                .into();
        }
    }

    let definitions = clippy_lints.into_iter().map(|clippy_lint| {
        let lint_ident = format!("clippy::{}", clippy_lint.id);
        let doc = clippy_lint.help;

        quote! { LintCompletion { label: #lint_ident, description: #doc } }
    });

    let ts = quote! {
        pub(super) const CLIPPY_LINTS:  &[LintCompletion] = &[
            #(#definitions),*
        ];
    };

    Ok(ts)
}
