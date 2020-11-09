include ../../run-make-fulldeps/tools.mk

# FIXME https://github.com/rust-lang/rust/issues/78911
# ignore-32bit wrong/no cross compiler and sometimes we pass wrong gcc args (-m64)

# Tests that we don't ICE during incremental compilation after modifying a
# function span such that its previous end line exceeds the number of lines
# in the new file, but its start line/column and length remain the same.

SRC=$(TMPDIR)/src
INCR=$(TMPDIR)/incr

all:
	mkdir $(SRC)
	mkdir $(INCR)
	cp a.rs $(SRC)/main.rs
	$(RUSTC) -C incremental=$(INCR) $(SRC)/main.rs
	cp b.rs $(SRC)/main.rs
	$(RUSTC) -C incremental=$(INCR) $(SRC)/main.rs
