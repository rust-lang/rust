# needs-sanitizer-support
# needs-sanitizer-memory

# This test builds a trivial program with memory sanitizer
# and rebuilds part of the standard library

# -Clink-dead-code is a setting that sanitizer front-ends generally enable,
# due to issues with certain linkers mangling sections used to track
# coverage when trying to remove dead code
# Linking dead code allows tracking more regressions generating code for the
# standard library, eg for target features that are not currently enabled.

# -Ccodegen-units=1 is also enabled by the front-ends, to work around
# misoptimisations with ThinLTO.

all:
	RUSTFLAGS="-Cpasses=sancov -Clink-dead-code -Zsanitizer=memory -Ccodegen-units=1" \
	cargo build --verbose --release -Zbuild-std=core --target $(TARGET)

