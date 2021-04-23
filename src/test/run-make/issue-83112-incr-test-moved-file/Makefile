include ../../run-make-fulldeps/tools.mk

# ignore-none no-std is not supported
# ignore-nvptx64-nvidia-cuda FIXME: can't find crate for 'std'

# Regression test for issue #83112
# The generated test harness code contains spans with a dummy location,
# but a non-dummy SyntaxContext. Previously, the incremental cache was encoding
# these spans as a full span (with a source file index), instead of skipping
# the encoding of the location information. If the file gest moved, the hash
# of the span will be unchanged (since it has a dummy location), so the incr
# cache would end up try to load a non-existent file using the previously
# enccoded source file id.

SRC=$(TMPDIR)/src
INCR=$(TMPDIR)/incr

all:
	mkdir $(SRC)
	mkdir $(SRC)/mydir
	mkdir $(INCR)
	cp main.rs $(SRC)/main.rs
	$(RUSTC) --test -C incremental=$(INCR) $(SRC)/main.rs --target $(TARGET)
	mv $(SRC)/main.rs $(SRC)/mydir/main.rs
	$(RUSTC) --test -C incremental=$(INCR) $(SRC)/mydir/main.rs --target $(TARGET)
