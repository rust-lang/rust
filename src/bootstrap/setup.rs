use crate::t;
use std::path::{Path, PathBuf};
use std::{
    env, fs,
    io::{self, Write},
};

pub fn setup(src_path: &Path, include_name: &str) {
    let cfg_file = env::var_os("BOOTSTRAP_CONFIG").map(PathBuf::from);

    if cfg_file.as_ref().map_or(false, |f| f.exists()) {
        let file = cfg_file.unwrap();
        println!(
            "error: you asked `x.py` to setup a new config file, but one already exists at `{}`",
            file.display()
        );
        println!(
            "help: try adding `profile = \"{}\"` at the top of {}",
            include_name,
            file.display()
        );
        println!(
            "note: this will use the configuration in {}/src/bootstrap/defaults/config.{}.toml",
            src_path.display(),
            include_name
        );
        std::process::exit(1);
    }

    let path = cfg_file.unwrap_or_else(|| src_path.join("config.toml"));
    let settings = format!(
        "# Includes one of the default files in src/bootstrap/defaults\n\
    profile = \"{}\"\n",
        include_name
    );
    t!(fs::write(path, settings));

    let include_path =
        format!("{}/src/bootstrap/defaults/config.{}.toml", src_path.display(), include_name);
    println!("`x.py` will now use the configuration at {}", include_path);

    let suggestions = match include_name {
        "codegen" | "compiler" => &["check", "build", "test"][..],
        "library" => &["check", "build", "test library/std", "doc"],
        "user" => &["dist", "build"],
        _ => return,
    };

    println!("To get started, try one of the following commands:");
    for cmd in suggestions {
        println!("- `x.py {}`", cmd);
    }

    if include_name != "user" {
        println!(
            "For more suggestions, see https://rustc-dev-guide.rust-lang.org/building/suggested.html"
        );
    }
}

// Used to get the path for `Subcommand::Setup`
pub fn interactive_path() -> io::Result<String> {
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
        break match input.trim().to_lowercase().as_str() {
            "a" | "lib" | "library" => "library",
            "b" | "compiler" => "compiler",
            "c" | "llvm" => "llvm",
            "d" | "user" | "maintainer" => "maintainer",
            _ => {
                println!("error: unrecognized option '{}'", input.trim());
                println!("note: press Ctrl+C to exit");
                continue;
            }
        };
    };
    Ok(template.to_owned())
}
