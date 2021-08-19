-include ../tools.mk

# Here's the basic overview: the crates all form a chain of dependencies,
#
#   m6 -> m5 -> m4 -> m3 -> m2 -> m1
#
# but: some crates have both rlibs and dylibs available, and others have only
# rlib, or only dylib.
#
#   m6 (app) -> m5 (rlib) -> m4 (dylib) -> m3 (rlib,dylib) -> m2 (lib) -> m1 (dylib,rlib)
#
# So the question we explore in this test is: What is the effect of `-C
# prefer-dynamic` at different points in the chain.
#
# For example, if you try to build the above without *any* use of `-C
# prefer-dynamic`, build of `m6` fails, because the compilation of dylib for
# `m4` prefers the rlib for `m3` (which itself statically links the rlib for
# `m1`). This is illustrated in v1 below.

RLIB_TYPE=--crate-type rlib --cfg rlib_type
DYLIB_TYPE=--crate-type dylib --cfg dylib_type
LIB_TYPE=--crate-type lib --cfg lib_type
BIN_TYPE=--crate-type bin --cfg bin_type
PREFER_DYNAMIC_INPUT=-Zprefer-dynamic-subset -Cprefer-dynamic=std,m2,m3,$(1)

# How to read this command sequence, for people unversed in shell scripting:
#
# I have tried to write useful comments as `echo` statements. Their text is usually
# delimited by single quotes ('') rathern than double quotes ("") so that I can use
# backticks (``) in the notes without it being interpreted as a shell invocation.
#
# `COMMAND && exit 1 && exit 0` is how we run a command and say that we're
# expecting it to fail.
#
# `>&2 echo "stuff"` emits "stuff" to stderr; that's useful for humans reviewing
# the error output on expected errors.
#
# (I'm not going so far as to programmatically verify the error output actually
# matches our expectations here; that is a bridge too far. Instead, I'm just
# trusting that if an error arises, its for the reason stated in the comment
# emitted to stderr.)
#
# Note: you can add a `false` to the end of the recipe if you want to forcibly
# see the stderr and stdout output captured in `compiletest`.

all:
	echo 'v1: no extra flags, just see what default behavior is for this strange chain of crates.'
	>&2 echo '`m1` build as dylib and rlib, v1'
	$(RUSTC) m1.rs $(DYLIB_TYPE)
	$(RUSTC) m1.rs $(RLIB_TYPE)
	>&2 echo '`m2` build as lib, v1'
	$(RUSTC) m2.rs $(LIB_TYPE)
	>&2 echo '`m3` build as dylib, v1; no prefer-dynamic: expected to select rlib for `m1`'
	$(RUSTC) m3.rs $(DYLIB_TYPE)
	>&2 echo '`m3` build as rlib, v1'
	$(RUSTC) m3.rs $(RLIB_TYPE)
	>&2 echo '`m4` build as dylib, v1; no prefer-dynamic: expected to select rlib for `m3`'
	$(RUSTC) m4.rs $(DYLIB_TYPE)
	>&2 echo '`m5` build as rlib, v1'
	$(RUSTC) m5.rs $(RLIB_TYPE)
	>&2 echo '`m6` build binary, v1: selects dylib for `m4`, because no rlib available'
	$(RUSTC) m6.rs $(BIN_TYPE) && exit 1 || exit 0
	>&2 echo 'above expected to fail: `m1`, `m3` and `std` linked dynamically via `m6` and statically via `m4`.'

	echo 'v2: most immediate fix for problem from previous version: prefer-dynamic on `m4`.'
	>&2 echo '`m4` build as dylib v2: prefer-dynamic (selects dylibs for `std` and `m3`)'
	$(RUSTC) m4.rs $(DYLIB_TYPE) -C prefer-dynamic
	>&2 echo '`m5` build as rlib v2'
	$(RUSTC) m5.rs $(RLIB_TYPE)
	>&2 echo '`m6` build binary, v2: selects dylib for `m4`, because no rlib available'
	$(RUSTC) m6.rs $(BIN_TYPE)
	echo 'run `m6` v2'
	$(call RUN,m6) > $(TMPDIR)/v2.output
	echo 'expectation: prefer-dynamic on `m4` selected dylib for `m3`, while latter linked rlib for `m1`.'
	echo 'linkage: m6:bin m5:rlib m4:dylib m3:dylib m2:lib m1:rlib' > $(TMPDIR)/v2.expect
	diff -u $(TMPDIR)/v2.output $(TMPDIR)/v2.expect
	echo 'expectation: removing m4 dylib will cause run attempt to fail.'
	>&2 echo 'remove m4 dylib'
	$(call REMOVE_DYLIBS,m4)
	$(call FAIL,m6)
	>&2 echo 'rebuild m4 and check that recovered state.'
	$(RUSTC) m4.rs $(DYLIB_TYPE) -C prefer-dynamic
	$(call RUN,m6)
	echo 'expectation: removing m3 dylib alone will cause run attempt to fail.'
	>&2 echo 'remove m3 dylib'
	$(call REMOVE_DYLIBS,m3)
	$(call FAIL,m6)
	>&2 echo 'rebuild m3 and check that recovered state.'
	$(RUSTC) m3.rs $(DYLIB_TYPE)
	$(call RUN,m6)
	echo 'expectation: removing m1 dylib and rlib will not perturb run v2 attempt.'
	>&2 echo 'remove `m1` rlib and dylib'
	$(call REMOVE_DYLIBS,m1)
	$(call REMOVE_RLIBS,m1)
	$(call RUN,m6)
	>&2 echo 'rebuild `m1` rlib and dylib'
	$(RUSTC) m1.rs $(RLIB_TYPE)
	$(RUSTC) m1.rs $(DYLIB_TYPE)

	echo 'v3: try `-C prefer-dynamic` when compiling `m2` now.'
	>&2 echo 'rebuild m1 dylib'
	$(RUSTC) m1.rs $(RLIB_TYPE)
	$(RUSTC) m1.rs $(DYLIB_TYPE)
	>&2 echo '`m2` build as lib, v3: prefer-dynamic (overall, still selects m1.rlib)'
	$(RUSTC) m2.rs $(LIB_TYPE) -C prefer-dynamic
	>&2 echo '`m3` build as dylib, v3'
	$(RUSTC) m3.rs $(DYLIB_TYPE)
	>&2 echo '`m3` build as rlib, v3'
	$(RUSTC) m3.rs $(RLIB_TYPE)
	>&2 echo '`m4` build as dylib v3: prefer-dynamic (to avoid duplicate static linkage)'
	$(RUSTC) m4.rs $(DYLIB_TYPE) -C prefer-dynamic
	>&2 echo '`m5` build as rlib v3'
	$(RUSTC) m5.rs $(RLIB_TYPE)
	>&2 echo '`m6` build bin v3'
	$(RUSTC) m6.rs $(BIN_TYPE)
	$(call RUN,m6) > $(TMPDIR)/v3.output
	echo 'expectation: prefer-dynamic on `m2` had no effect on which m1 got linked compared to previous.'
	echo 'linkage: m6:bin m5:rlib m4:dylib m3:dylib m2:lib m1:rlib' > $(TMPDIR)/v3.expect
	diff -u $(TMPDIR)/v3.output $(TMPDIR)/v3.expect

	echo 'v4: try `-C prefer-dynamic` when compiling `m3` now.'
	>&2 echo 'rebuild m1 dylib'
	$(RUSTC) m1.rs $(RLIB_TYPE)
	$(RUSTC) m1.rs $(DYLIB_TYPE)
	>&2 echo '`m2` build as lib, v4'
	$(RUSTC) m2.rs $(LIB_TYPE)
	>&2 echo '`m3` build as dylib, v4; prefer-dynamic, and now selects m1.dylib'
	$(RUSTC) m3.rs $(DYLIB_TYPE) -C prefer-dynamic
	>&2 echo '`m3` build as rlib, v4: prefer-dynamic'
	$(RUSTC) m3.rs $(RLIB_TYPE) -C prefer-dynamic
	>&2 echo '`m4` build as dylib v4'
	$(RUSTC) m4.rs $(DYLIB_TYPE)
	>&2 echo '`m5` build as rlib v4'
	$(RUSTC) m5.rs $(RLIB_TYPE)
	>&2 echo '`m6` build bin v4: selects dylib for `m4`, because no rlib available'
	$(RUSTC) m6.rs $(BIN_TYPE) && exit 1 || exit 0
	>&2 echo 'above expected to fail: `std` linked dynamically via `m6` and statically via `m4`.'

	echo 'v5: try `-C prefer-dynamic` when compiling `m3` now.'
	>&2 echo 'rebuild m1 dylib'
	$(RUSTC) m1.rs $(RLIB_TYPE)
	$(RUSTC) m1.rs $(DYLIB_TYPE)
	>&2 echo '`m2` build as lib, v5'
	$(RUSTC) m2.rs $(LIB_TYPE)
	>&2 echo '`m3` build as dylib, v5; prefer-dynamic, and now selects dylib for `m1`.'
	$(RUSTC) m3.rs $(DYLIB_TYPE) -C prefer-dynamic
	>&2 echo '`m3` build as rlib, v5: prefer-dynamic; it nonetheless resolves to rlib for `m1`'
	$(RUSTC) m3.rs $(RLIB_TYPE) -C prefer-dynamic
	>&2 echo '`m4` build as dylib v5; prefer-dynamic just for std, not m3'
	$(RUSTC) m4.rs $(DYLIB_TYPE) -C prefer-dynamic=std -Z prefer-dynamic-std
	>&2 echo '`m5` build as rlib v5'
	$(RUSTC) m5.rs $(RLIB_TYPE)
	>&2 echo '`m6` build bin v5; prefer-dynamic just for `m4` and `std`, not `m3`'
	$(RUSTC) m6.rs $(BIN_TYPE) -C prefer-dynamic=std,m4 -Z prefer-dynamic-subset
	$(call RUN,m6) > $(TMPDIR)/v5.output
	echo 'expectation: prefer-dynamic=std,m4 on `m4`/`m6` meant m3 rlib gets linked, and m3 rlib links m1 rlib.'
	echo 'linkage: m6:bin m5:rlib m4:dylib m3:rlib m2:lib m1:rlib' > $(TMPDIR)/v5.expect
	diff -u $(TMPDIR)/v5.output $(TMPDIR)/v5.expect

	echo 'v6: try `-C prefer-dynamic` when `m3` and `m4` now.'
	>&2 echo '`m4` build as dylib v6; prefer-dynamic for everything; selects dylib for `m3`.'
	$(RUSTC) m4.rs $(DYLIB_TYPE) -C prefer-dynamic
	>&2 echo '`m5` build as rlib v6'
	$(RUSTC) m5.rs $(RLIB_TYPE)
	>&2 echo '`m6` build bin v6: selects dylib for `m4` and thus for `m3` and `m1`.'
	$(RUSTC) m6.rs $(BIN_TYPE)
	$(call RUN,m6) > $(TMPDIR)/v6.output
	echo 'expectation: prefer-dynamic on `m3`/`m4` meant m3 dylib gets linked, and m3 dylib links m1 dylib.'
	echo 'linkage: m6:bin m5:rlib m4:dylib m3:dylib m2:lib m1:dylib' > $(TMPDIR)/v6.expect
	diff -u $(TMPDIR)/v6.output $(TMPDIR)/v6.expect

#	>&2 echo 'reached final false'
#	false

# Note: you can add a `false` to the end of the recipe if you want to forcibly
# see the stderr and stdout output captured in `compiletest`.
