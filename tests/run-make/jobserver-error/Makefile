include ../../run-make-fulldeps/tools.mk

# only-linux

# Test compiler behavior in case: `jobserver-auth` points to correct pipe which is not jobserver.

all:
	bash -c 'echo "fn main() {}" | MAKEFLAGS="--jobserver-auth=3,3" $(RUSTC) - 3</dev/null' 2>&1 | diff jobserver.stderr -
