extern crate gcc;

fn main() {
	let sources = &[
		"../rt/miniz.c",
	];

	gcc::compile_library("libminiz.a", sources);
}
