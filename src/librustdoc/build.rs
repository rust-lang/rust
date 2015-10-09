extern crate gcc;

fn main() {
	let sources = &[
		"../rt/hoedown/src/autolink.c",
		"../rt/hoedown/src/buffer.c",
		"../rt/hoedown/src/document.c",
		"../rt/hoedown/src/escape.c",
		"../rt/hoedown/src/html.c",
		"../rt/hoedown/src/html_blocks.c",
		"../rt/hoedown/src/html_smartypants.c",
		"../rt/hoedown/src/stack.c",
		"../rt/hoedown/src/version.c",
	];

	gcc::compile_library("libhoedown.a", sources);

	println!("cargo:rustc-cfg=rustdoc")
}
