use std::process::Command;
use std::path::PathBuf;
use std::env;

extern crate gcc;

fn main() {
	let dir = env::current_dir().unwrap();
	let llvmdeps = PathBuf::from(&env::var_os("CFG_LLVM_LINKAGE_FILE").unwrap());
	let llvm_config = PathBuf::from(&env::var_os("CFG_LLVM_CONFIG").unwrap());
	let std_cpp = PathBuf::from(&env::var_os("CFG_LLVM_STDCPP").unwrap());

	let sources = &[
		"../rustllvm/ArchiveWrapper.cpp",
		"../rustllvm/ExecutionEngineWrapper.cpp",
		"../rustllvm/PassWrapper.cpp",
		"../rustllvm/RustWrapper.cpp",
	];

	let mklldeps = dir.join("../etc/mklldeps.py");

	run(Command::new("python2")
		.arg(mklldeps)
		.arg(llvmdeps)
		.arg("")
		.arg("1")
		.arg(&llvm_config)
		.arg(std_cpp));

	println!("cargo:rustc-link-lib=ffi");
	println!("cargo:rustc-link-search=native={}", String::from_utf8(Command::new(llvm_config).arg("--libdir").output().unwrap().stdout).unwrap());

	let mut config = gcc::Config::new();
	config.flag("-std=c++11");
	compile_library("librustllvm.a", sources, config);
}

fn run(cmd: &mut Command) {
	println!("running: {:?}", cmd);
	cmd.status().unwrap();
}

fn compile_library(output: &str, files: &[&str], mut c: gcc::Config) {
	for f in files.iter() {
		c.file(*f);
	}
	c.compile(output)
}
