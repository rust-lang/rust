include ../tools.mk

# only-linux
# ignore-test: This test randomly fails, see https://github.com/rust-lang/rust/issues/110321

# Test compiler behavior in case: `jobserver-auth` points to correct pipe which is not jobserver.

all:
	bash -c 'echo "fn main() {}" | MAKEFLAGS="--jobserver-auth=3,3" $(RUSTC) - 3</dev/null' 2>&1 | diff jobserver.stderr -
