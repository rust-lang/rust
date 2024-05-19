# When provided standard input piped directly into rustc, this test checks that the compilation completes successfully and that the output can be executed.
# See https://github.com/rust-lang/rust/pull/28805

# ignore-cross-compile
include ../tools.mk

all:
	echo 'fn main(){}' | $(RUSTC) -
	$(call RUN,rust_out)
