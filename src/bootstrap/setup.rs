use crate::{t, VERSION};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
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
    User,
}

impl Profile {
    fn include_path(&self, src_path: &Path) -> PathBuf {
        PathBuf::from(format!("{}/src/bootstrap/defaults/config.{}.toml", src_path.display(), self))
    }

    pub fn all() -> impl Iterator<Item = Self> {
        use Profile::*;
        // N.B. these are ordered by how they are displayed, not alphabetically
        [Library, Compiler, Codegen, User].iter().copied()
    }

    pub fn purpose(&self) -> String {
        use Profile::*;
        match self {
            Library => "Contribute to the standard library",
            Compiler => "Contribute to the compiler or rustdoc",
            Codegen => "Contribute to the compiler, and also modify LLVM or codegen",
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
            "compiler" | "rustdoc" => Ok(Profile::Compiler),
            "llvm" | "codegen" => Ok(Profile::Codegen),
            "maintainer" | "user" => Ok(Profile::User),
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

    let path = cfg_file.unwrap_or_else(|| src_path.join("config.toml"));
    let settings = format!(
        "# Includes one of the default files in src/bootstrap/defaults\n\
    profile = \"{}\"\n\
    changelog-seen = {}\n",
        profile, VERSION
    );
    t!(fs::write(path, settings));

    let include_path = profile.include_path(src_path);
    println!("`x.py` will now use the configuration at {}", include_path.display());

    let suggestions = match profile {
        Profile::Codegen | Profile::Compiler => &["check", "build", "test"][..],
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

// Used to get the path for `Subcommand::Setup`
pub fn interactive_path() -> io::Result<Profile> {
    fn abbrev_all() -> impl Iterator<Item = (String, Profile)> {
        ('a'..).map(|c| c.to_string()).zip(Profile::all())
    }

    fn parse_with_abbrev(input: &str) -> Result<Profile, String> {
        let input = input.trim().to_lowercase();
        for (letter, profile) in abbrev_all() {
            if input == letter {
                return Ok(profile);
            }
        }
        input.parse()
    }

    println!("Welcome to the Rust project! What do you want to do with x.py?");
    for (letter, profile) in abbrev_all() {
        println!("{}) {}: {}", letter, profile, profile.purpose());
    }
    let template = loop {
        print!(
            "Please choose one ({}): ",
            abbrev_all().map(|(l, _)| l).collect::<Vec<_>>().join("/")
        );
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input == "" {
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
If you'd like, x.py can install a git hook for you that will automatically run `tidy --bless` on each commit
to ensure your code is up to par. If you decide later that this behavior is undesirable,
simply delete the `pre-commit` file from .git/hooks."
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

    Ok(if should_install {
        let src = src_path.join("src").join("etc").join("pre-commit.sh");
        let git = t!(Command::new("git").args(&["rev-parse", "--git-common-dir"]).output().map(
            |output| {
                assert!(output.status.success(), "failed to run `git`");
                PathBuf::from(t!(String::from_utf8(output.stdout)).trim())
            }
        ));
        let dst = git.join("hooks").join("pre-commit");
        match fs::hard_link(src, &dst) {
            Err(e) => println!(
                "error: could not create hook {}: do you already have the git hook installed?\n{}",
                dst.display(),
                e
            ),
            Ok(_) => println!("Linked `src/etc/pre-commit.sh` to `.git/hooks/pre-commit`"),
        };
    } else {
        println!("Ok, skipping installation!");
    })
}
