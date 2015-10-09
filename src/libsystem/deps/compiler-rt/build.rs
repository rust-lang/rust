use std::process::Command;
use std::path::PathBuf;
use std::fs::{copy, rename};
use std::env;

extern crate gcc;

fn main() {
    let dir = env::current_dir().unwrap();
    let out_dir = PathBuf::from(&env::var_os("OUT_DIR").unwrap());

    let compiler_rt = dir.join("../../../compiler-rt");

    let config = gcc::Config::new().get_compiler();
    let cflags = config.args().into_iter().cloned().map(|c| c.into_string().unwrap()).fold(String::new(), |mut args, arg| { args.push_str(&arg); args.push(' '); args });

    run(Command::new("make")
        .current_dir(&out_dir)
        .arg("-C").arg(&compiler_rt)
        .arg(format!("ProjSrcRoot={}", compiler_rt.display()))
        .arg(format!("ProjObjRoot={}", out_dir.display()))
        .arg(format!("TargetTriple={}", env::var("TARGET").unwrap()))
        .arg(format!("CFLAGS={}", cflags))
        .arg(format!("CC={}", config.path().display()))
        .arg("triple-builtins"));

    copy(&out_dir.join("triple").join("builtins").join("libcompiler_rt.a"), &out_dir.join("../../../deps/libcompiler-rt.a")).unwrap();
    rename(&out_dir.join("triple").join("builtins").join("libcompiler_rt.a"), &out_dir.join("libcompiler_rt.a")).unwrap();

    println!("cargo:rustc-flags=-L native={} -l static={}", out_dir.display(), "compiler_rt");
}

fn run(cmd: &mut Command) {
    println!("running: {:?}", cmd);
    cmd.status().unwrap();
}
