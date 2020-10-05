use crate::t;
use std::path::{Path, PathBuf};
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
        [Profile::Compiler, Profile::Codegen, Profile::Library, Profile::User].iter().copied()
    }
}

#[derive(Debug)]
pub struct ProfileErr {
    pub name: String,
}

impl FromStr for Profile {
    type Err = ProfileErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "a" | "lib" | "library" => Ok(Profile::Library),
            "b" | "compiler" => Ok(Profile::Compiler),
            "c" | "llvm" | "codegen" => Ok(Profile::Codegen),
            "d" | "maintainer" | "user" => Ok(Profile::User),
            _ => Err(ProfileErr { name: s.to_string() }),
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
    profile = \"{}\"\n",
        profile
    );
    t!(fs::write(path, settings));

    let include_path = profile.include_path(src_path);
    println!("`x.py` will now use the configuration at {}", include_path.display());

    let suggestions = match profile {
        Profile::Codegen | Profile::Compiler => &["check", "build", "test"][..],
        Profile::Library => &["check", "build", "test library/std", "doc"],
        Profile::User => &["dist", "build"],
    };

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
    let mut input = String::new();
    println!(
        "Welcome to the Rust project! What do you want to do with x.py?
a) Contribute to the standard library
b) Contribute to the compiler
c) Contribute to the compiler, and also modify LLVM or codegen
d) Install Rust from source"
    );
    let template = loop {
        print!("Please choose one (a/b/c/d): ");
        io::stdout().flush()?;
        io::stdin().read_line(&mut input)?;
        break match input.trim().to_lowercase().parse() {
            Ok(profile) => profile,
            Err(ProfileErr { name }) => {
                println!("error: unrecognized option '{}'", name);
                println!("note: press Ctrl+C to exit");
                continue;
            }
        };
    };
    Ok(template)
}
