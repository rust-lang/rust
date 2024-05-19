include ../tools.mk

# ignore-none no-std is not supported
# ignore-nvptx64-nvidia-cuda FIXME: can't find crate for 'std'

# Ensure that modifying a crate on disk (without recompiling it)
# does not cause ICEs in downstream crates.
# Previously, we would call `SourceMap.guess_head_span` on a span
# from an external crate, which would cause us to read an upstream
# source file from disk during compilation of a downstream crate
# See #86480 for more details

INCR=$(TMPDIR)/incr

all:
	cp first_crate.rs second_crate.rs $(TMPDIR)
	$(RUSTC) $(TMPDIR)/first_crate.rs  -C incremental=$(INCR) --target $(TARGET) --crate-type lib
	$(RUSTC) $(TMPDIR)/second_crate.rs -C incremental=$(INCR) --target $(TARGET)  --extern first_crate=$(TMPDIR)/libfirst_crate.rlib --crate-type lib
	rm $(TMPDIR)/first_crate.rs
	$(RUSTC) $(TMPDIR)/second_crate.rs  -C incremental=$(INCR) --target $(TARGET) --cfg second_run --crate-type lib

