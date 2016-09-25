-include ../tools.mk
all: main.rs
	echo '[{"path": "/foo/foo.rs", "text": "pub fn foo() -> bool { true }\n"}]' | $(RUSTC) main.rs -Zreplace_files
