use crate::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::Config;
use crate::{t, VERSION};
use sha2::Digest;
use std::env::consts::EXE_SUFFIX;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf, MAIN_SEPARATOR};
use std::process::Command;
use std::str::FromStr;
use std::{fmt, fs, io};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Profile {
    Compiler,
    Codegen,
    Library,
    Tools,
    Dist,
    None,
}

/// A list of historical hashes of `src/etc/rust_analyzer_settings.json`.
/// New entries should be appended whenever this is updated so we can detect
/// outdated vs. user-modified settings files.
static SETTINGS_HASHES: &[&str] = &[
    "ea67e259dedf60d4429b6c349a564ffcd1563cf41c920a856d1f5b16b4701ac8",
    "56e7bf011c71c5d81e0bf42e84938111847a810eee69d906bba494ea90b51922",
    "af1b5efe196aed007577899db9dae15d6dbc923d6fa42fa0934e68617ba9bbe0",
    "3468fea433c25fff60be6b71e8a215a732a7b1268b6a83bf10d024344e140541",
];
static RUST_ANALYZER_SETTINGS: &str = include_str!("../etc/rust_analyzer_settings.json");

impl Profile {
    fn include_path(&self, src_path: &Path) -> PathBuf {
        PathBuf::from(format!("{}/src/bootstrap/defaults/config.{}.toml", src_path.display(), self))
    }

    pub fn all() -> impl Iterator<Item = Self> {
        use Profile::*;
        // N.B. these are ordered by how they are displayed, not alphabetically
        [Library, Compiler, Codegen, Tools, Dist, None].iter().copied()
    }

    pub fn purpose(&self) -> String {
        use Profile::*;
        match self {
            Library => "Contribute to the standard library",
            Compiler => "Contribute to the compiler itself",
            Codegen => "Contribute to the compiler, and also modify LLVM or codegen",
            Tools => "Contribute to tools which depend on the compiler, but do not modify it directly (e.g. rustdoc, clippy, miri)",
            Dist => "Install Rust from source",
            None => "Do not modify `config.toml`"
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
            Profile::Codegen => "codegen",
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
            "llvm" | "codegen" => Ok(Profile::Codegen),
            "maintainer" | "dist" => Ok(Profile::Dist),
            "tools" | "tool" | "rustdoc" | "clippy" | "miri" | "rustfmt" | "rls" => {
                Ok(Profile::Tools)
            }
            "none" => Ok(Profile::None),
            _ => Err(format!("unknown profile: '{}'", s)),
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
        setup(&builder.build.config, self)
    }
}

pub fn setup(config: &Config, profile: Profile) {
    let suggestions: &[&str] = match profile {
        Profile::Codegen | Profile::Compiler | Profile::None => &["check", "build", "test"],
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
        println!("- `x.py {}`", cmd);
    }

    if profile != Profile::Dist {
        println!(
            "For more suggestions, see https://rustc-dev-guide.rust-lang.org/building/suggested.html"
        );
    }

    let path = &config.config.clone().unwrap_or(PathBuf::from("config.toml"));
    setup_config_toml(path, profile, config);
}

fn setup_config_toml(path: &PathBuf, profile: Profile, config: &Config) {
    if profile == Profile::None {
        return;
    }
    if path.exists() {
        eprintln!();
        eprintln!(
            "error: you asked `x.py` to setup a new config file, but one already exists at `{}`",
            path.display()
        );
        eprintln!("help: try adding `profile = \"{}\"` at the top of {}", profile, path.display());
        eprintln!(
            "note: this will use the configuration in {}",
            profile.include_path(&config.src).display()
        );
        crate::detail_exit_macro!(1);
    }

    let settings = format!(
        "# Includes one of the default files in src/bootstrap/defaults\n\
    profile = \"{}\"\n\
    changelog-seen = {}\n",
        profile, VERSION
    );

    t!(fs::write(path, settings));

    let include_path = profile.include_path(&config.src);
    println!("`x.py` will now use the configuration at {}", include_path.display());
}

/// Creates a toolchain link for stage1 using `rustup`
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
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
        if let [cmd] = &run.paths[..] {
            if cmd.assert_single_path().path.as_path().as_os_str() == "link" {
                run.builder.ensure(Link);
            }
        }
    }
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let config = &builder.config;
        if config.dry_run() {
            return;
        }
        let stage_path =
            ["build", config.build.rustc_target_arg(), "stage1"].join(&MAIN_SEPARATOR.to_string());

        if !rustup_installed() {
            eprintln!("`rustup` is not installed; cannot link `stage1` toolchain");
        } else if stage_dir_exists(&stage_path[..]) && !config.dry_run() {
            attempt_toolchain_link(&stage_path[..]);
        }
    }
}

fn rustup_installed() -> bool {
    Command::new("rustup")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .output()
        .map_or(false, |output| output.status.success())
}

fn stage_dir_exists(stage_path: &str) -> bool {
    match fs::create_dir(&stage_path) {
        Ok(_) => true,
        Err(_) => Path::new(&stage_path).exists(),
    }
}

fn attempt_toolchain_link(stage_path: &str) {
    if toolchain_is_linked() {
        return;
    }

    if !ensure_stage1_toolchain_placeholder_exists(stage_path) {
        eprintln!(
            "Failed to create a template for stage 1 toolchain or confirm that it already exists"
        );
        return;
    }

    if try_link_toolchain(&stage_path) {
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

fn toolchain_is_linked() -> bool {
    match Command::new("rustup")
        .args(&["toolchain", "list"])
        .stdout(std::process::Stdio::piped())
        .output()
    {
        Ok(toolchain_list) => {
            if !String::from_utf8_lossy(&toolchain_list.stdout).contains("stage1") {
                return false;
            }
            // The toolchain has already been linked.
            println!(
                "`stage1` toolchain already linked; not attempting to link `stage1` toolchain"
            );
        }
        Err(_) => {
            // In this case, we don't know if the `stage1` toolchain has been linked;
            // but `rustup` failed, so let's not go any further.
            println!(
                "`rustup` failed to list current toolchains; not attempting to link `stage1` toolchain"
            );
        }
    }
    true
}

fn try_link_toolchain(stage_path: &str) -> bool {
    Command::new("rustup")
        .stdout(std::process::Stdio::null())
        .args(&["toolchain", "link", "stage1", &stage_path])
        .output()
        .map_or(false, |output| output.status.success())
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

    let pathbuf = pathbuf.join(format!("rustc{}", EXE_SUFFIX));

    if pathbuf.exists() {
        return true;
    }

    // Take care not to overwrite the file
    let result = File::options().append(true).create(true).open(&pathbuf);
    if result.is_err() {
        return false;
    }

    return true;
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
            crate::detail_exit_macro!(1);
        }
        break match parse_with_abbrev(&input) {
            Ok(profile) => profile,
            Err(err) => {
                eprintln!("error: {}", err);
                eprintln!("note: press Ctrl+C to exit");
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
                eprintln!("error: unrecognized option '{}'", input.trim());
                eprintln!("note: press Ctrl+C to exit");
            }
        };
    }
}

/// Installs `src/etc/pre-push.sh` as a Git hook
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Hook;

impl Step for Hook {
    type Output = ();
    const DEFAULT: bool = true;
    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("hook")
    }
    fn make_run(run: RunConfig<'_>) {
        if run.builder.config.dry_run() {
            return;
        }
        if let [cmd] = &run.paths[..] {
            if cmd.assert_single_path().path.as_path().as_os_str() == "hook" {
                run.builder.ensure(Hook);
            }
        }
    }
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let config = &builder.config;
        if config.dry_run() {
            return;
        }
        t!(install_git_hook_maybe(&config));
    }
}

// install a git hook to automatically run tidy, if they want
fn install_git_hook_maybe(config: &Config) -> io::Result<()> {
    let git = t!(config.git().args(&["rev-parse", "--git-common-dir"]).output().map(|output| {
        assert!(output.status.success(), "failed to run `git`");
        PathBuf::from(t!(String::from_utf8(output.stdout)).trim())
    }));
    let dst = git.join("hooks").join("pre-push");
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
    let src = config.src.join("src").join("etc").join("pre-push.sh");
    match fs::hard_link(src, &dst) {
        Err(e) => {
            eprintln!(
                "error: could not create hook {}: do you already have the git hook installed?\n{}",
                dst.display(),
                e
            );
            return Err(e);
        }
        Ok(_) => println!("Linked `src/etc/pre-push.sh` to `.git/hooks/pre-push`"),
    };
    Ok(())
}

/// Sets up or displays `src/etc/rust_analyzer_settings.json`
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Vscode;

impl Step for Vscode {
    type Output = ();
    const DEFAULT: bool = true;
    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("vscode")
    }
    fn make_run(run: RunConfig<'_>) {
        if run.builder.config.dry_run() {
            return;
        }
        if let [cmd] = &run.paths[..] {
            if cmd.assert_single_path().path.as_path().as_os_str() == "vscode" {
                run.builder.ensure(Vscode);
            }
        }
    }
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let config = &builder.config;
        if config.dry_run() {
            return;
        }
        t!(create_vscode_settings_maybe(&config));
    }
}

/// Create a `.vscode/settings.json` file for rustc development, or just print it
fn create_vscode_settings_maybe(config: &Config) -> io::Result<()> {
    let (current_hash, historical_hashes) = SETTINGS_HASHES.split_last().unwrap();
    let vscode_settings = config.src.join(".vscode").join("settings.json");
    // If None, no settings.json exists
    // If Some(true), is a previous version of settings.json
    // If Some(false), is not a previous version (i.e. user modified)
    // If it's up to date we can just skip this
    let mut mismatched_settings = None;
    if let Ok(current) = fs::read_to_string(&vscode_settings) {
        let mut hasher = sha2::Sha256::new();
        hasher.update(&current);
        let hash = hex::encode(hasher.finalize().as_slice());
        if hash == *current_hash {
            return Ok(());
        } else if historical_hashes.contains(&hash.as_str()) {
            mismatched_settings = Some(true);
        } else {
            mismatched_settings = Some(false);
        }
    }
    println!(
        "\nx.py can automatically install the recommended `.vscode/settings.json` file for rustc development"
    );
    match mismatched_settings {
        Some(true) => eprintln!(
            "warning: existing `.vscode/settings.json` is out of date, x.py will update it"
        ),
        Some(false) => eprintln!(
            "warning: existing `.vscode/settings.json` has been modified by user, x.py will back it up and replace it"
        ),
        _ => (),
    }
    let should_create = match prompt_user(
        "Would you like to create/update `settings.json`, or only print suggested settings?: [y/p/N]",
    )? {
        Some(PromptResult::Yes) => true,
        Some(PromptResult::Print) => false,
        _ => {
            println!("Ok, skipping settings!");
            return Ok(());
        }
    };
    if should_create {
        let path = config.src.join(".vscode");
        if !path.exists() {
            fs::create_dir(&path)?;
        }
        let verb = match mismatched_settings {
            // exists but outdated, we can replace this
            Some(true) => "Updated",
            // exists but user modified, back it up
            Some(false) => {
                // exists and is not current version or outdated, so back it up
                let mut backup = vscode_settings.clone();
                backup.set_extension("bak");
                eprintln!("warning: copying `settings.json` to `settings.json.bak`");
                fs::copy(&vscode_settings, &backup)?;
                "Updated"
            }
            _ => "Created",
        };
        fs::write(&vscode_settings, &RUST_ANALYZER_SETTINGS)?;
        println!("{verb} `.vscode/settings.json`");
    } else {
        println!("\n{RUST_ANALYZER_SETTINGS}");
    }
    Ok(())
}
