use std::collections::BTreeMap;
use std::io::Write;
use std::ops::ControlFlow::*;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::{fs, io};

use serde_derive::{Deserialize, Serialize};
use sha2::Digest;

use crate::core::build_steps::format;
use crate::core::build_steps::setup::prompt_user;
use crate::core::builder::{Builder, CommandLineStep, RunConfig, ShouldRun};
use crate::core::config::Config;
use crate::utils::helpers::{hex_encode, t};

#[cfg(test)]
mod tests;

/// Handles editor-specific setup differences
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum EditorKind {
    Emacs,
    Helix,
    Vim,
    VsCode,
    Zed,
}

static PARSED_HASHES: LazyLock<BTreeMap<EditorKind, Vec<&'static str>>> = LazyLock::new(|| {
    const ALL_HASHES: &str = include_str!("hashes.json");
    let mut map: BTreeMap<_, Vec<_>> = serde_json::from_str(ALL_HASHES).unwrap();
    map.insert(EditorKind::Vim, map.get(&EditorKind::VsCode).unwrap().clone());
    map
});

impl EditorKind {
    // Used in `./tests.rs`.
    #[cfg(test)]
    pub const ALL: &[EditorKind] = &[
        EditorKind::Emacs,
        EditorKind::Helix,
        EditorKind::Vim,
        EditorKind::VsCode,
        EditorKind::Zed,
    ];

    fn prompt_user() -> io::Result<Option<EditorKind>> {
        let prompt_str = "Available editors:
1. Emacs
2. Helix
3. Vim
4. VS Code
5. Zed

Select which editor you would like to set up [default: None]: ";

        let mut input = String::new();
        loop {
            print!("{prompt_str}");
            io::stdout().flush()?;
            io::stdin().read_line(&mut input)?;

            let mut modified_input = input.to_lowercase();
            modified_input.retain(|ch| !ch.is_whitespace());
            match modified_input.as_str() {
                "1" | "emacs" => return Ok(Some(EditorKind::Emacs)),
                "2" | "helix" => return Ok(Some(EditorKind::Helix)),
                "3" | "vim" => return Ok(Some(EditorKind::Vim)),
                "4" | "vscode" => return Ok(Some(EditorKind::VsCode)),
                "5" | "zed" => return Ok(Some(EditorKind::Zed)),
                "" | "none" => return Ok(None),
                _ => {
                    eprintln!("ERROR: unrecognized option '{}'", input.trim());
                    eprintln!("NOTE: press Ctrl+C to exit");
                }
            }

            input.clear();
        }
    }

    /// A list of historical hashes of each LSP settings file
    /// New entries should be appended whenever this is updated so we can detect
    /// outdated vs. user-modified settings files.
    fn hashes(&self) -> &'static [&'static str] {
        PARSED_HASHES.get(self).unwrap()
    }

    fn settings_path(&self, config: &Config) -> PathBuf {
        config.src.join(self.settings_short_path())
    }

    fn settings_short_path(&self) -> PathBuf {
        self.settings_folder().join(match self {
            EditorKind::Emacs => ".dir-locals.el",
            EditorKind::Helix => "languages.toml",
            EditorKind::Vim => "coc-settings.json",
            EditorKind::VsCode | EditorKind::Zed => "settings.json",
        })
    }

    fn settings_folder(&self) -> PathBuf {
        match self {
            EditorKind::Emacs => PathBuf::new(),
            EditorKind::Helix => PathBuf::from(".helix"),
            EditorKind::Vim => PathBuf::from(".vim"),
            EditorKind::VsCode => PathBuf::from(".vscode"),
            EditorKind::Zed => PathBuf::from(".zed"),
        }
    }

    fn settings_template(&self) -> &'static str {
        match self {
            EditorKind::Emacs => include_str!("../../../../../etc/rust_analyzer_eglot.el"),
            EditorKind::Helix => include_str!("../../../../../etc/rust_analyzer_helix.toml"),
            EditorKind::Vim | EditorKind::VsCode => {
                include_str!("../../../../../etc/rust_analyzer_settings.json")
            }
            EditorKind::Zed => include_str!("../../../../../etc/rust_analyzer_zed.json"),
        }
    }

    fn backup_extension(&self) -> String {
        format!("{}.bak", self.settings_short_path().extension().unwrap().to_str().unwrap())
    }
}

/// Sets up or displays the LSP config for one of the supported editors
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Editor;

impl CommandLineStep for Editor {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("editor")
    }

    fn is_default_step(_builder: &Builder<'_>) -> bool {
        true
    }

    fn make_run(run: RunConfig<'_>) {
        if run.builder.config.dry_run() {
            return;
        }
        if let [cmd] = &run.paths[..]
            && cmd.assert_single_path().path.as_path().as_os_str() == "editor"
        {
            run.builder.ensure(Editor);
        }
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let config = &builder.config;
        if config.dry_run() {
            return;
        }
        match EditorKind::prompt_user() {
            Ok(editor_kind) => {
                if let Some(editor_kind) = editor_kind {
                    while !t!(create_editor_settings_maybe(config, &editor_kind)) {}

                    // Also pre-download stage 0 rustfmt, so that the IDE configs which point to
                    // `build/host/rustfmt` have an available binary to work with.
                    builder.ensure(format::InternalRustfmt);
                } else {
                    println!("Ok, skipping editor setup!");
                }
            }
            Err(e) => eprintln!("Could not determine the editor: {e}"),
        }
    }
}

/// Create the recommended editor LSP config file for rustc development, or just print it
/// If this method should be re-called, it returns `false`.
fn create_editor_settings_maybe(config: &Config, editor: &EditorKind) -> io::Result<bool> {
    let hashes = editor.hashes();
    let (current_hash, historical_hashes) = hashes.split_last().unwrap();
    let settings_path = editor.settings_path(config);
    let settings_short_path = editor.settings_short_path();
    let settings_filename = settings_short_path.to_str().unwrap();
    // If None, no settings file exists
    // If Some(true), is a previous version of settings.json
    // If Some(false), is not a previous version (i.e. user modified)
    // If it's up to date we can just skip this
    let mut mismatched_settings = None;
    if let Ok(current) = fs::read_to_string(&settings_path) {
        let mut hasher = sha2::Sha256::new();
        hasher.update(&current);
        let hash = hex_encode(hasher.finalize().as_slice());
        if hash == *current_hash {
            return Ok(true);
        } else if historical_hashes.contains(&hash.as_str()) {
            mismatched_settings = Some(true);
        } else {
            mismatched_settings = Some(false);
        }
    }
    println!(
        "\nx.py can automatically install the recommended `{settings_filename}` file for rustc development"
    );

    match mismatched_settings {
        Some(true) => {
            eprintln!("WARNING: existing `{settings_filename}` is out of date, x.py will update it")
        }
        Some(false) => eprintln!(
            "WARNING: existing `{settings_filename}` has been modified by user, x.py will back it up and replace it"
        ),
        _ => (),
    }

    enum Action {
        Create,
        Print,
        Exit,
    }

    let action = prompt_user(
        &format!(
            "Would you like to create/update `{settings_filename}`? (Press 'p' to preview values): [y/p/N]"
        ),
        |input| match input {
            "y" => Break(Action::Create),
            "p" => Break(Action::Print),
            "n" | "" => Break(Action::Exit),
            _ => Continue(()),
        },
    )?;

    match action {
        Action::Create => {
            let settings_folder_path = config.src.join(editor.settings_folder());
            if !settings_folder_path.exists() {
                fs::create_dir(settings_folder_path)?;
            }
            let verb = match mismatched_settings {
                // exists but outdated, we can replace this
                Some(true) => "Updated",
                // exists but user modified, back it up
                Some(false) => {
                    // exists and is not current version or outdated, so back it up
                    let backup = settings_path.with_extension(editor.backup_extension());
                    eprintln!(
                        "WARNING: copying `{}` to `{}`",
                        settings_path.file_name().unwrap().to_str().unwrap(),
                        backup.file_name().unwrap().to_str().unwrap(),
                    );
                    fs::copy(&settings_path, &backup)?;
                    "Updated"
                }
                _ => "Created",
            };
            fs::write(&settings_path, editor.settings_template())?;
            println!("{verb} `{settings_filename}`");
            Ok(true)
        }
        Action::Print => {
            println!("\n{}", editor.settings_template());
            Ok(false)
        }
        Action::Exit => {
            println!("Ok, skipping settings!");
            return Ok(true);
        }
    }
}
