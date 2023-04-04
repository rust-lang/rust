include ../tools.mk

INCR=$(TMPDIR)/incr
FIRST_SRC=$(TMPDIR)/first_src
SECOND_SRC=$(TMPDIR)/second_src

# ignore-none no-std is not supported
# ignore-nvptx64-nvidia-cuda FIXME: can't find crate for 'std'

# Tests that we don't get an ICE when the working directory
# (but not the build directory!) changes between compilation
# sessions

all:
	mkdir $(INCR)
	# Build from 'FIRST_SRC'
	mkdir $(FIRST_SRC)
	cp my_lib.rs $(FIRST_SRC)/my_lib.rs
	cp main.rs $(FIRST_SRC)/main.rs
	cd $(FIRST_SRC) && \
		$(RUSTC) -C incremental=$(INCR) --crate-type lib my_lib.rs --target $(TARGET) && \
		$(RUSTC) -C incremental=$(INCR) --extern my_lib=$(TMPDIR)/libmy_lib.rlib main.rs --target $(TARGET)
	# Build from 'SECOND_SRC', keeping the output directory and incremental directory
	# the same
	mv $(FIRST_SRC) $(SECOND_SRC)
	cd $(SECOND_SRC) && \
		$(RUSTC) -C incremental=$(INCR) --crate-type lib my_lib.rs --target $(TARGET) && \
		$(RUSTC) -C incremental=$(INCR) --extern my_lib=$(TMPDIR)/libmy_lib.rlib main.rs --target $(TARGET)
