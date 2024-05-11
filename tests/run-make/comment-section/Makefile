# Both GCC and Clang write by default a `.comment` section with compiler information. Rustc received a similar .comment section, so this tests checks that this section properly appears.
# See https://github.com/rust-lang/rust/commit/74b8d324eb77a8f337b35dc68ac91b0c2c06debc

include ../tools.mk

# only-linux

all:
	echo 'fn main(){}' | $(RUSTC) - --emit=link,obj -Csave-temps --target=$(TARGET)

	# Check linked output has a `.comment` section with the expected content.
	readelf -p '.comment' $(TMPDIR)/rust_out | $(CGREP) -F 'rustc version 1.'

	# Check all object files (including temporary outputs) have a `.comment`
	# section with the expected content.
	set -e; for f in $(TMPDIR)/*.o; do \
		readelf -p '.comment' $$f | $(CGREP) -F 'rustc version 1.'; \
	done
