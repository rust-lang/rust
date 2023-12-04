include ../tools.mk

# only-linux
# ignore-cross-compile

# Test compiler behavior in case environment specifies wrong jobserver.

all:
	bash -c 'echo "fn main() {}" | MAKEFLAGS="--jobserver-auth=3,3" $(RUSTC)' 2>&1 | diff cannot_open_fd.stderr -
	bash -c 'echo "fn main() {}" | MAKEFLAGS="--jobserver-auth=3,3" $(RUSTC) - 3</dev/null' 2>&1 | diff not_a_pipe.stderr -

# This test randomly fails, see https://github.com/rust-lang/rust/issues/110321
disabled:
	bash -c 'echo "fn main() {}" | MAKEFLAGS="--jobserver-auth=3,3" $(RUSTC) - 3< <(cat /dev/null)' 2>&1 | diff poisoned_pipe.stderr -

