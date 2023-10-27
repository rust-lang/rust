use std::env;
use std::process;

mod build;
mod config;
mod prepare;
mod rustc_info;
mod test;
mod utils;

macro_rules! arg_error {
    ($($err:tt)*) => {{
        eprintln!($($err)*);
        eprintln!();
        usage();
        std::process::exit(1);
    }};
}

fn usage() {
    println!(
        "\
Available commands for build_system:

    prepare  : Run prepare command
    build    : Run build command
    test     : Run test command
    --help   : Show this message"
    );
}

pub enum Command {
    Prepare,
    Build,
    Test,
}

fn main() {
    if env::var("RUST_BACKTRACE").is_err() {
        env::set_var("RUST_BACKTRACE", "1");
    }

    let command = match env::args().nth(1).as_deref() {
        Some("prepare") => Command::Prepare,
        Some("build") => Command::Build,
        Some("test") => Command::Test,
        Some("--help") => {
            usage();
            process::exit(0);
        }
        Some(flag) if flag.starts_with('-') => arg_error!("Expected command found flag {}", flag),
        Some(command) => arg_error!("Unknown command {}", command),
        None => {
            usage();
            process::exit(0);
        }
    };

    if let Err(e) = match command {
        Command::Prepare => prepare::run(),
        Command::Build => build::run(),
        Command::Test => test::run(),
    } {
        eprintln!("Command failed to run: {e:?}");
        process::exit(1);
    }
}
