use crate::TargetSelection;
use crate::{t, VERSION};
use std::env::consts::EXE_SUFFIX;
use std::fmt::Write as _;
use std::fs::File;
use std::path::{Path, PathBuf, MAIN_SEPARATOR};
use std::process::Command;
use std::str::FromStr;
use std::{
    env, fmt, fs,
    io::{self, Write},
};

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Profile {
    Compiler,
    Codegen,
    Library,
    Tools,
    User,
}

impl Profile {
    fn include_path(&self, src_path: &Path) -> PathBuf {
        PathBuf::from(format!("{}/src/bootstrap/defaults/config.{}.toml", src_path.display(), self))
    }

    pub fn all() -> impl Iterator<Item = Self> {
        use Profile::*;
        // N.B. these are ordered by how they are displayed, not alphabetically
        [Library, Compiler, Codegen, Tools, User].iter().copied()
    }

    pub fn purpose(&self) -> String {
        use Profile::*;
        match self {
            Library => "Contribute to the standard library",
            Compiler => "Contribute to the compiler itself",
            Codegen => "Contribute to the compiler, and also modify LLVM or codegen",
            Tools => "Contribute to tools which depend on the compiler, but do not modify it directly (e.g. rustdoc, clippy, miri)",
            User => "Install Rust from source",
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
}

impl FromStr for Profile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "lib" | "library" => Ok(Profile::Library),
            "compiler" => Ok(Profile::Compiler),
            "llvm" | "codegen" => Ok(Profile::Codegen),
            "maintainer" | "user" => Ok(Profile::User),
            "tools" | "tool" | "rustdoc" | "clippy" | "miri" | "rustfmt" | "rls" => {
                Ok(Profile::Tools)
            }
            _ => Err(format!("unknown profile: '{}'", s)),
        }
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Profile::Compiler => write!(f, "compiler"),
            Profile::Codegen => write!(f, "codegen"),
            Profile::Library => write!(f, "library"),
            Profile::User => write!(f, "user"),
            Profile::Tools => write!(f, "tools"),
        }
    }
}

pub fn setup(src_path: &Path, profile: Profile) {
    let cfg_file = env::var_os("BOOTSTRAP_CONFIG").map(PathBuf::from);

    if cfg_file.as_ref().map_or(false, |f| f.exists()) {
        let file = cfg_file.unwrap();
        println!(
            "error: you asked `x.py` to setup a new config file, but one already exists at `{}`",
            file.display()
        );
        println!("help: try adding `profile = \"{}\"` at the top of {}", profile, file.display());
        println!(
            "note: this will use the configuration in {}",
            profile.include_path(src_path).display()
        );
        std::process::exit(1);
    }

    let path = cfg_file.unwrap_or_else(|| "config.toml".into());
    let settings = format!(
        "# Includes one of the default files in src/bootstrap/defaults\n\
    profile = \"{}\"\n\
    changelog-seen = {}\n",
        profile, VERSION
    );
    t!(fs::write(path, settings));

    let include_path = profile.include_path(src_path);
    println!("`x.py` will now use the configuration at {}", include_path.display());

    let build = TargetSelection::from_user(&env!("BUILD_TRIPLE"));
    let stage_path =
        ["build", build.rustc_target_arg(), "stage1"].join(&MAIN_SEPARATOR.to_string());

    println!();

    if !rustup_installed() && profile != Profile::User {
        println!("`rustup` is not installed; cannot link `stage1` toolchain");
    } else if stage_dir_exists(&stage_path[..]) {
        attempt_toolchain_link(&stage_path[..]);
    }

    let suggestions = match profile {
        Profile::Codegen | Profile::Compiler => &["check", "build", "test"][..],
        Profile::Tools => &[
            "check",
            "build",
            "test src/test/rustdoc*",
            "test src/tools/clippy",
            "test src/tools/miri",
            "test src/tools/rustfmt",
        ],
        Profile::Library => &["check", "build", "test library/std", "doc"],
        Profile::User => &["dist", "build"],
    };

    println!();

    t!(install_git_hook_maybe(src_path));

    println!();

    println!("To get started, try one of the following commands:");
    for cmd in suggestions {
        println!("- `x.py {}`", cmd);
    }

    if profile != Profile::User {
        println!(
            "For more suggestions, see https://rustc-dev-guide.rust-lang.org/building/suggested.html"
        );
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
    match fs::create_dir(&stage_path[..]) {
        Ok(_) => true,
        Err(_) => Path::new(&stage_path[..]).exists(),
    }
}

fn attempt_toolchain_link(stage_path: &str) {
    if toolchain_is_linked() {
        return;
    }

    if !ensure_stage1_toolchain_placeholder_exists(stage_path) {
        println!(
            "Failed to create a template for stage 1 toolchain or confirm that it already exists"
        );
        return;
    }

    if try_link_toolchain(&stage_path[..]) {
        println!(
            "Added `stage1` rustup toolchain; try `cargo +stage1 build` on a separate rust project to run a newly-built toolchain"
        );
    } else {
        println!("`rustup` failed to link stage 1 build to `stage1` toolchain");
        println!(
            "To manually link stage 1 build to `stage1` toolchain, run:\n
            `rustup toolchain link stage1 {}`",
            &stage_path[..]
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
        .args(&["toolchain", "link", "stage1", &stage_path[..]])
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
            std::process::exit(1);
        }
        break match parse_with_abbrev(&input) {
            Ok(profile) => profile,
            Err(err) => {
                println!("error: {}", err);
                println!("note: press Ctrl+C to exit");
                continue;
            }
        };
    };
    Ok(template)
}

// install a git hook to automatically run tidy --bless, if they want
fn install_git_hook_maybe(src_path: &Path) -> io::Result<()> {
    let mut input = String::new();
    println!(
        "Rust's CI will automatically fail if it doesn't pass `tidy`, the internal tool for ensuring code quality.
If you'd like, x.py can install a git hook for you that will automatically run `tidy --bless` before
pushing your code to ensure your code is up to par. If you decide later that this behavior is
undesirable, simply delete the `pre-push` file from .git/hooks."
    );

    let should_install = loop {
        print!("Would you like to install the git hook?: [y/N] ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        break match input.trim().to_lowercase().as_str() {
            "y" | "yes" => true,
            "n" | "no" | "" => false,
            _ => {
                println!("error: unrecognized option '{}'", input.trim());
                println!("note: press Ctrl+C to exit");
                continue;
            }
        };
    };

    if should_install {
        let src = src_path.join("src").join("etc").join("pre-push.sh");
        let git = t!(Command::new("git").args(&["rev-parse", "--git-common-dir"]).output().map(
            |output| {
                assert!(output.status.success(), "failed to run `git`");
                PathBuf::from(t!(String::from_utf8(output.stdout)).trim())
            }
        ));
        let dst = git.join("hooks").join("pre-push");
        match fs::hard_link(src, &dst) {
            Err(e) => println!(
                "error: could not create hook {}: do you already have the git hook installed?\n{}",
                dst.display(),
                e
            ),
            Ok(_) => println!("Linked `src/etc/pre-commit.sh` to `.git/hooks/pre-push`"),
        };
    } else {
        println!("Ok, skipping installation!");
    }
    Ok(())
}
