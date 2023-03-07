# ignore-none no-std is not supported
# ignore-nvptx64-nvidia-cuda FIXME: can't find crate for `std`

include ../../run-make-fulldeps/tools.mk

# Tests that we don't ICE during incremental compilation after modifying a
# function span such that its previous end line exceeds the number of lines
# in the new file, but its start line/column and length remain the same.

SRC=$(TMPDIR)/src
INCR=$(TMPDIR)/incr

all:
	mkdir $(SRC)
	mkdir $(INCR)
	cp a.rs $(SRC)/main.rs
	$(RUSTC) -C incremental=$(INCR) $(SRC)/main.rs --target $(TARGET)
	cp b.rs $(SRC)/main.rs
	$(RUSTC) -C incremental=$(INCR) $(SRC)/main.rs --target $(TARGET)
