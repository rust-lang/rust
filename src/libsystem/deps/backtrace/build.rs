use std::process::Command;
use std::io::Write;
use std::fs::{ rename, OpenOptions };
use std::path::PathBuf;
use std::env;

extern crate gcc;

fn main() {
    let dir = env::current_dir().unwrap();
    let out_dir = PathBuf::from(&env::var_os("OUT_DIR").unwrap());

    let config = gcc::Config::new().get_compiler();
    let cflags = config.args().into_iter().cloned().map(|c| c.into_string().unwrap()).fold(String::new(), |mut args, arg| { args.push_str(&arg); args.push(' '); args });

    let backtrace = dir.join("../../../libbacktrace");

    run(Command::new("sh")
        .env("CFLAGS", cflags)
        .env("CC", config.path())
        .current_dir(&out_dir)
        .arg(backtrace.join("configure"))
        .arg("--with-pic")
        .arg(format!("--host={}", env::var("TARGET").unwrap()))
        .arg(format!("--build={}", env::var("HOST").unwrap())));

    let config = out_dir.join("config.h");
    let mut config = OpenOptions::new().append(true).write(true).open(&config).unwrap();
    config.write_all("\n#undef HAVE_ATOMIC_FUNCTIONS\n#undef HAVE_SYNC_FUNCTIONS\n".as_bytes()).unwrap();
    drop(config);

    run(Command::new("make")
        .current_dir(&out_dir)
        .arg(&format!("-j{}", env::var("NUM_JOBS").unwrap()))
        .arg(&format!("INCDIR={}", backtrace.display())));

    rename(&out_dir.join(".libs").join("libbacktrace.a"), &out_dir.join("libbacktrace.a")).unwrap();

    println!("cargo:rustc-flags=-L native={} -l static={}", out_dir.display(), "backtrace");
}

fn run(cmd: &mut Command) {
    println!("running: {:?}", cmd);
    cmd.status().unwrap();
}
