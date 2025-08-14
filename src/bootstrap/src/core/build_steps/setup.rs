//! First time setup of a dev environment
//!
//! These are build-and-run steps for `./x.py setup`, which allows quickly setting up the directory
//! for modifying, building, and running the compiler and library. Running arbitrary configuration
//! allows setting up things that cannot be simply captured inside the bootstrap.toml, in addition to
//! leading people away from manually editing most of the bootstrap.toml values.

use std::env::consts::EXE_SUFFIX;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write;
use std::path::{MAIN_SEPARATOR_STR, Path, PathBuf};
use std::str::FromStr;
use std::{fmt, fs, io};

use sha2::Digest;

use crate::core::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::utils::change_tracker::CONFIG_CHANGE_HISTORY;
use crate::utils::exec::command;
use crate::utils::helpers::{self, hex_encode};
use crate::{Config, t};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Profile {
    Compiler,
    Library,
    Tools,
    Dist,
    None,
}

static PROFILE_DIR: &str = "src/bootstrap/defaults";

impl Profile {
    fn include_path(&self, src_path: &Path) -> PathBuf {
        PathBuf::from(format!("{}/{PROFILE_DIR}/bootstrap.{}.toml", src_path.display(), self))
    }

    pub fn all() -> impl Iterator<Item = Self> {
        use Profile::*;
        // N.B. these are ordered by how they are displayed, not alphabetically
        [Library, Compiler, Tools, Dist, None].iter().copied()
    }

    pub fn purpose(&self) -> String {
        use Profile::*;
        match self {
            Library => "Contribute to the standard library",
            Compiler => "Contribute to the compiler itself",
            Tools => "Contribute to tools which depend on the compiler, but do not modify it directly (e.g. rustdoc, clippy, miri)",
            Dist => "Install Rust from source",
            None => "Do not modify `bootstrap.toml`"
        }
        .to_string()
    }

    pub fn all_for_help(indent: &str) -> String {
        let mut out = String::new();
        for choice in Profile::all() {
            writeln!(&mut out, "{}{}: {}", indent, choice, choice.purpose()).unwrap();
        }
        out
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Profile::Compiler => "compiler",
            Profile::Library => "library",
            Profile::Tools => "tools",
            Profile::Dist => "dist",
            Profile::None => "none",
        }
    }
}

impl FromStr for Profile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "lib" | "library" => Ok(Profile::Library),
            "compiler" => Ok(Profile::Compiler),
            "maintainer" | "dist" | "user" => Ok(Profile::Dist),
            "tools" | "tool" | "rustdoc" | "clippy" | "miri" | "rustfmt" => Ok(Profile::Tools),
            "none" => Ok(Profile::None),
            "llvm" | "codegen" => Err("the \"llvm\" and \"codegen\" profiles have been removed,\
                use \"compiler\" instead which has the same functionality"
                .to_string()),
            _ => Err(format!("unknown profile: '{s}'")),
        }
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Step for Profile {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(mut run: ShouldRun<'_>) -> ShouldRun<'_> {
        for choice in Profile::all() {
            run = run.alias(choice.as_str());
        }
        run
    }

    fn make_run(run: RunConfig<'_>) {
        if run.builder.config.dry_run() {
            return;
        }

        let path = &run.builder.config.config.clone().unwrap_or(PathBuf::from("bootstrap.toml"));
        if path.exists() {
            eprintln!();
            eprintln!(
                "ERROR: you asked for a new config file, but one already exists at `{}`",
                t!(path.canonicalize()).display()
            );

            match prompt_user(
                "Do you wish to override the existing configuration (which will allow the setup process to continue)?: [y/N]",
            ) {
                Ok(Some(PromptResult::Yes)) => {
                    t!(fs::remove_file(path));
                }
                _ => {
                    println!("Exiting.");
                    crate::exit!(1);
                }
            }
        }

        // for Profile, `run.paths` will have 1 and only 1 element
        // this is because we only accept at most 1 path from user input.
        // If user calls `x.py setup` without arguments, the interactive TUI
        // will guide user to provide one.
        let profile = if run.paths.len() > 1 {
            // HACK: `builder` runs this step with all paths if no path was passed.
            t!(interactive_path())
        } else {
            run.paths
                .first()
                .unwrap()
                .assert_single_path()
                .path
                .as_path()
                .as_os_str()
                .to_str()
                .unwrap()
                .parse()
                .unwrap()
        };

        run.builder.ensure(profile);
    }

    fn run(self, builder: &Builder<'_>) {
        setup(&builder.build.config, self);
    }
}

pub fn setup(config: &Config, profile: Profile) {
    let suggestions: &[&str] = match profile {
        Profile::Compiler | Profile::None => &["check", "build", "test"],
        Profile::Tools => &[
            "check",
            "build",
            "test tests/rustdoc*",
            "test src/tools/clippy",
            "test src/tools/miri",
            "test src/tools/rustfmt",
        ],
        Profile::Library => &["check", "build", "test library/std", "doc"],
        Profile::Dist => &["dist", "build"],
    };

    println!();

    println!("To get started, try one of the following commands:");
    for cmd in suggestions {
        println!("- `x.py {cmd}`");
    }

    if profile != Profile::Dist {
        println!(
            "For more suggestions, see https://rustc-dev-guide.rust-lang.org/building/suggested.html"
        );
    }

    if profile == Profile::Tools {
        eprintln!();
        eprintln!(
            "NOTE: the `tools` profile sets up the `stage2` toolchain (use \
            `rustup toolchain link 'name' build/host/stage2` to use rustc)"
        )
    }

    let path = &config.config.clone().unwrap_or(PathBuf::from("bootstrap.toml"));
    setup_config_toml(path, profile, config);
}

fn setup_config_toml(path: &Path, profile: Profile, config: &Config) {
    if profile == Profile::None {
        return;
    }

    let latest_change_id = CONFIG_CHANGE_HISTORY.last().unwrap().change_id;
    let settings = format!(
        "# See bootstrap.example.toml for documentation of available options\n\
    #\n\
    profile = \"{profile}\"  # Includes one of the default files in {PROFILE_DIR}\n\
    change-id = {latest_change_id}\n"
    );

    t!(fs::write(path, settings));

    let include_path = profile.include_path(&config.src);
    println!("`x.py` will now use the configuration at {}", include_path.display());
}

/// Creates a toolchain link for stage1 using `rustup`
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Link;
impl Step for Link {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("link")
    }

    fn make_run(run: RunConfig<'_>) {
        if run.builder.config.dry_run() {
            return;
        }
        if let [cmd] = &run.paths[..]
            && cmd.assert_single_path().path.as_path().as_os_str() == "link"
        {
            run.builder.ensure(Link);
        }
    }
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let config = &builder.config;

        if config.dry_run() {
            return;
        }

        if !rustup_installed(builder) {
            println!("WARNING: `rustup` is not installed; Skipping `stage1` toolchain linking.");
            return;
        }

        let stage_path =
            ["build", config.host_target.rustc_target_arg(), "stage1"].join(MAIN_SEPARATOR_STR);

        if stage_dir_exists(&stage_path[..]) && !config.dry_run() {
            attempt_toolchain_link(builder, &stage_path[..]);
        }
    }
}

fn rustup_installed(builder: &Builder<'_>) -> bool {
    let mut rustup = command("rustup");
    rustup.arg("--version");

    rustup.allow_failure().run_in_dry_run().run_capture_stdout(builder).is_success()
}

fn stage_dir_exists(stage_path: &str) -> bool {
    match fs::create_dir(stage_path) {
        Ok(_) => true,
        Err(_) => Path::new(&stage_path).exists(),
    }
}

fn attempt_toolchain_link(builder: &Builder<'_>, stage_path: &str) {
    if toolchain_is_linked(builder) {
        return;
    }

    if !ensure_stage1_toolchain_placeholder_exists(stage_path) {
        eprintln!(
            "Failed to create a template for stage 1 toolchain or confirm that it already exists"
        );
        return;
    }

    if try_link_toolchain(builder, stage_path) {
        println!(
            "Added `stage1` rustup toolchain; try `cargo +stage1 build` on a separate rust project to run a newly-built toolchain"
        );
    } else {
        eprintln!("`rustup` failed to link stage 1 build to `stage1` toolchain");
        eprintln!(
            "To manually link stage 1 build to `stage1` toolchain, run:\n
            `rustup toolchain link stage1 {}`",
            &stage_path
        );
    }
}

fn toolchain_is_linked(builder: &Builder<'_>) -> bool {
    match command("rustup")
        .allow_failure()
        .args(["toolchain", "list"])
        .run_capture_stdout(builder)
        .stdout_if_ok()
    {
        Some(toolchain_list) => {
            if !toolchain_list.contains("stage1") {
                return false;
            }
            // The toolchain has already been linked.
            println!(
                "`stage1` toolchain already linked; not attempting to link `stage1` toolchain"
            );
        }
        None => {
            // In this case, we don't know if the `stage1` toolchain has been linked;
            // but `rustup` failed, so let's not go any further.
            println!(
                "`rustup` failed to list current toolchains; not attempting to link `stage1` toolchain"
            );
        }
    }
    true
}

fn try_link_toolchain(builder: &Builder<'_>, stage_path: &str) -> bool {
    command("rustup")
        .args(["toolchain", "link", "stage1", stage_path])
        .run_capture_stdout(builder)
        .is_success()
}

fn ensure_stage1_toolchain_placeholder_exists(stage_path: &str) -> bool {
    let pathbuf = PathBuf::from(stage_path);

    if fs::create_dir_all(pathbuf.join("lib")).is_err() {
        return false;
    };

    let pathbuf = pathbuf.join("bin");
    if fs::create_dir_all(&pathbuf).is_err() {
        return false;
    };

    let pathbuf = pathbuf.join(format!("rustc{EXE_SUFFIX}"));

    if pathbuf.exists() {
        return true;
    }

    // Take care not to overwrite the file
    let result = File::options().append(true).create(true).open(&pathbuf);
    if result.is_err() {
        return false;
    }

    true
}

// Used to get the path for `Subcommand::Setup`
pub fn interactive_path() -> io::Result<Profile> {
    fn abbrev_all() -> impl Iterator<Item = ((String, String), Profile)> {
        ('a'..)
            .zip(1..)
            .map(|(letter, number)| (letter.to_string(), number.to_string()))
            .zip(Profile::all())
    }

    fn parse_with_abbrev(input: &str) -> Result<Profile, String> {
        let input = input.trim().to_lowercase();
        for ((letter, number), profile) in abbrev_all() {
            if input == letter || input == number {
                return Ok(profile);
            }
        }
        input.parse()
    }

    println!("Welcome to the Rust project! What do you want to do with x.py?");
    for ((letter, _), profile) in abbrev_all() {
        println!("{}) {}: {}", letter, profile, profile.purpose());
    }
    let template = loop {
        print!(
            "Please choose one ({}): ",
            abbrev_all().map(|((l, _), _)| l).collect::<Vec<_>>().join("/")
        );
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input.is_empty() {
            eprintln!("EOF on stdin, when expecting answer to question.  Giving up.");
            crate::exit!(1);
        }
        break match parse_with_abbrev(&input) {
            Ok(profile) => profile,
            Err(err) => {
                eprintln!("ERROR: {err}");
                eprintln!("NOTE: press Ctrl+C to exit");
                continue;
            }
        };
    };
    Ok(template)
}

#[derive(PartialEq)]
enum PromptResult {
    Yes,   // y/Y/yes
    No,    // n/N/no
    Print, // p/P/print
}

/// Prompt a user for a answer, looping until they enter an accepted input or nothing
fn prompt_user(prompt: &str) -> io::Result<Option<PromptResult>> {
    let mut input = String::new();
    loop {
        print!("{prompt} ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        match input.trim().to_lowercase().as_str() {
            "y" | "yes" => return Ok(Some(PromptResult::Yes)),
            "n" | "no" => return Ok(Some(PromptResult::No)),
            "p" | "print" => return Ok(Some(PromptResult::Print)),
            "" => return Ok(None),
            _ => {
                eprintln!("ERROR: unrecognized option '{}'", input.trim());
                eprintln!("NOTE: press Ctrl+C to exit");
            }
        };
    }
}

/// Installs `src/etc/pre-push.sh` as a Git hook
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Hook;

impl Step for Hook {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("hook")
    }

    fn make_run(run: RunConfig<'_>) {
        if let [cmd] = &run.paths[..]
            && cmd.assert_single_path().path.as_path().as_os_str() == "hook"
        {
            run.builder.ensure(Hook);
        }
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let config = &builder.config;

        if config.dry_run() || !config.rust_info.is_managed_git_subrepository() {
            return;
        }

        t!(install_git_hook_maybe(builder, config));
    }
}

// install a git hook to automatically run tidy, if they want
fn install_git_hook_maybe(builder: &Builder<'_>, config: &Config) -> io::Result<()> {
    let git = helpers::git(Some(&config.src))
        .args(["rev-parse", "--git-common-dir"])
        .run_capture(builder)
        .stdout();
    let git = PathBuf::from(git.trim());
    let hooks_dir = git.join("hooks");
    let dst = hooks_dir.join("pre-push");
    if dst.exists() {
        // The git hook has already been set up, or the user already has a custom hook.
        return Ok(());
    }

    println!(
        "\nRust's CI will automatically fail if it doesn't pass `tidy`, the internal tool for ensuring code quality.
If you'd like, x.py can install a git hook for you that will automatically run `test tidy` before
pushing your code to ensure your code is up to par. If you decide later that this behavior is
undesirable, simply delete the `pre-push` file from .git/hooks."
    );

    if prompt_user("Would you like to install the git hook?: [y/N]")? != Some(PromptResult::Yes) {
        println!("Ok, skipping installation!");
        return Ok(());
    }
    if !hooks_dir.exists() {
        // We need to (try to) create the hooks directory first.
        let _ = fs::create_dir(hooks_dir);
    }
    let src = config.src.join("src").join("etc").join("pre-push.sh");
    match fs::hard_link(src, &dst) {
        Err(e) => {
            eprintln!(
                "ERROR: could not create hook {}: do you already have the git hook installed?\n{}",
                dst.display(),
                e
            );
            return Err(e);
        }
        Ok(_) => println!("Linked `src/etc/pre-push.sh` to `.git/hooks/pre-push`"),
    };
    Ok(())
}

/// Handles editor-specific setup differences
#[derive(Clone, Debug, Eq, PartialEq)]
enum EditorKind {
    Emacs,
    Helix,
    Vim,
    VsCode,
    Zed,
}

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
        match self {
            EditorKind::Emacs => &[
                "51068d4747a13732440d1a8b8f432603badb1864fa431d83d0fd4f8fa57039e0",
                "d29af4d949bbe2371eac928a3c31cf9496b1701aa1c45f11cd6c759865ad5c45",
                "b5dd299b93dca3ceeb9b335f929293cb3d4bf4977866fbe7ceeac2a8a9f99088",
                "631c837b0e98ae35fd48b0e5f743b1ca60adadf2d0a2b23566ba25df372cf1a9",
                "080955765db84bb6cbf178879f489c4e2369397626a6ecb3debedb94a9d0b3ce",
                "f501475c6654187091c924ae26187fa5791d74d4a8ab3fb61fbbe4c0275aade1",
            ],
            EditorKind::Helix => &[
                "2d3069b8cf1b977e5d4023965eb6199597755e6c96c185ed5f2854f98b83d233",
                "6736d61409fbebba0933afd2e4c44ff2f97c1cb36cf0299a7f4a7819b8775040",
                "f252dcc30ca85a193a699581e5e929d5bd6c19d40d7a7ade5e257a9517a124a5",
                "198c195ed0c070d15907b279b8b4ea96198ca71b939f5376454f3d636ab54da5",
                "1c43ead340b20792b91d02b08494ee68708e7e09f56b6766629b4b72079208f1",
            ],
            EditorKind::Vim | EditorKind::VsCode => &[
                "ea67e259dedf60d4429b6c349a564ffcd1563cf41c920a856d1f5b16b4701ac8",
                "56e7bf011c71c5d81e0bf42e84938111847a810eee69d906bba494ea90b51922",
                "af1b5efe196aed007577899db9dae15d6dbc923d6fa42fa0934e68617ba9bbe0",
                "3468fea433c25fff60be6b71e8a215a732a7b1268b6a83bf10d024344e140541",
                "47d227f424bf889b0d899b9cc992d5695e1b78c406e183cd78eafefbe5488923",
                "b526bd58d0262dd4dda2bff5bc5515b705fb668a46235ace3e057f807963a11a",
                "828666b021d837a33e78d870b56d34c88a5e2c85de58b693607ec574f0c27000",
                "811fb3b063c739d261fd8590dd30242e117908f5a095d594fa04585daa18ec4d",
                "4eecb58a2168b252077369da446c30ed0e658301efe69691979d1ef0443928f4",
                "c394386e6133bbf29ffd32c8af0bb3d4aac354cba9ee051f29612aa9350f8f8d",
                "e53e9129ca5ee5dcbd6ec8b68c2d87376474eb154992deba3c6d9ab1703e0717",
                "f954316090936c7e590c253ca9d524008375882fa13c5b41d7e2547a896ff893",
                "701b73751efd7abd6487f2c79348dab698af7ac4427b79fa3d2087c867144b12",
                "a61df796c0c007cb6512127330564e49e57d558dec715703916a928b072a1054",
            ],
            EditorKind::Zed => &[
                "bbce727c269d1bd0c98afef4d612eb4ce27aea3c3a8968c5f10b31affbc40b6c",
                "a5380cf5dd9328731aecc5dfb240d16dac46ed272126b9728006151ef42f5909",
                "2e96bf0d443852b12f016c8fc9840ab3d0a2b4fe0b0fb3a157e8d74d5e7e0e26",
                "4fadd4c87389a601a27db0d3d74a142fa3a2e656ae78982e934dbe24bee32ad6",
                "f0bb3d23ab1a49175ab0ef5c4071af95bb03d01d460776cdb716d91333443382",
            ],
        }
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
            EditorKind::Emacs => include_str!("../../../../etc/rust_analyzer_eglot.el"),
            EditorKind::Helix => include_str!("../../../../etc/rust_analyzer_helix.toml"),
            EditorKind::Vim | EditorKind::VsCode => {
                include_str!("../../../../etc/rust_analyzer_settings.json")
            }
            EditorKind::Zed => include_str!("../../../../etc/rust_analyzer_zed.json"),
        }
    }

    fn backup_extension(&self) -> String {
        format!("{}.bak", self.settings_short_path().extension().unwrap().to_str().unwrap())
    }
}

/// Sets up or displays the LSP config for one of the supported editors
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Editor;

impl Step for Editor {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("editor")
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
    let should_create = match prompt_user(&format!(
        "Would you like to create/update `{settings_filename}`? (Press 'p' to preview values): [y/N]"
    ))? {
        Some(PromptResult::Yes) => true,
        Some(PromptResult::Print) => false,
        _ => {
            println!("Ok, skipping settings!");
            return Ok(true);
        }
    };
    if should_create {
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
    } else {
        println!("\n{}", editor.settings_template());
    }
    Ok(should_create)
}
