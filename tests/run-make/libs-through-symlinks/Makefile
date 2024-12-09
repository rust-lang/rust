# ignore-cross-compile
include ../tools.mk

# ignore-windows

# The option -n for the AIX ln command has a different purpose than it does
# on Linux. On Linux, the -n option is used to treat the destination path as
# normal file if it is a symbolic link to a directory, which is the default
# behavior of the AIX ln command.
ifeq ($(UNAME),AIX)
LN_FLAGS := -sf
else
LN_FLAGS := -nsf
endif

NAME := $(shell $(RUSTC) --print file-names foo.rs)

all:
	mkdir -p $(TMPDIR)/outdir
	$(RUSTC) foo.rs -o $(TMPDIR)/outdir/$(NAME)
	ln $(LN_FLAGS) outdir/$(NAME) $(TMPDIR)
	RUSTC_LOG=rustc_metadata::loader $(RUSTC) bar.rs
