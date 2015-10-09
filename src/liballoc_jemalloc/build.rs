use std::process::Command;
use std::fs::rename;
use std::path::PathBuf;
use std::env;

extern crate gcc;

fn main() {
	let dir = env::current_dir().unwrap();
	let out_dir = PathBuf::from(&env::var_os("OUT_DIR").unwrap());

	let config = gcc::Config::new().get_compiler();
	let cflags = config.args().into_iter().cloned().map(|c| c.into_string().unwrap()).fold(String::new(), |mut args, arg| { args.push_str(&arg); args.push(' '); args });

	let jemalloc = dir.join("../jemalloc");

	run(Command::new("sh")
		.env("CFLAGS", cflags)
		.env("CC", config.path())
		.current_dir(&out_dir)
		.arg(jemalloc.join("configure"))
		.arg("--with-jemalloc-prefix=je_")
		.arg("--disable-fill")
		.arg(format!("--host={}", env::var("TARGET").unwrap()))
		.arg(format!("--build={}", env::var("HOST").unwrap())));

	run(Command::new("make")
		.current_dir(&out_dir)
		.arg(&format!("-j{}", env::var("NUM_JOBS").unwrap()))
		.arg(&format!("INCDIR={}", jemalloc.display())));

	rename(&out_dir.join("lib").join("libjemalloc_pic.a"), &out_dir.join("libjemalloc.a")).unwrap();

	println!("cargo:rustc-flags=-L native={} -l static={}", out_dir.display(), "jemalloc");
}

fn run(cmd: &mut Command) {
	println!("running: {:?}", cmd);
	cmd.status().unwrap();
}
