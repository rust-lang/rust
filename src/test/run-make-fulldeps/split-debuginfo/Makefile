-include ../tools.mk

all: off packed unpacked

ifeq ($(UNAME),Darwin)
# If disabled, don't run dsymutil
off:
	rm -rf $(TMPDIR)/*.dSYM
	$(RUSTC) foo.rs -g -C split-debuginfo=off
	[ ! -d $(TMPDIR)/foo.dSYM ]

# Packed by default, but only if debuginfo is requested
packed:
	rm -rf $(TMPDIR)/*.dSYM
	$(RUSTC) foo.rs
	[ ! -d $(TMPDIR)/foo.dSYM ]
	rm -rf $(TMPDIR)/*.dSYM
	$(RUSTC) foo.rs -g
	[ -d $(TMPDIR)/foo.dSYM ]
	rm -rf $(TMPDIR)/*.dSYM
	$(RUSTC) foo.rs -g -C split-debuginfo=packed
	[ -d $(TMPDIR)/foo.dSYM ]
	rm -rf $(TMPDIR)/*.dSYM

# Object files are preserved with unpacked and `dsymutil` isn't run
unpacked:
	$(RUSTC) foo.rs -g -C split-debuginfo=unpacked
	ls $(TMPDIR)/*.o
	[ ! -d $(TMPDIR)/foo.dSYM ]
else
ifdef IS_WINDOWS
# Windows only supports =off
off:
packed:
unpacked:
else
# If disabled, don't run dsymutil
off:
	$(RUSTC) foo.rs -g -C split-debuginfo=off -Z unstable-options
	[ ! -f $(TMPDIR)/*.dwp ]
	[ ! -f $(TMPDIR)/*.dwo ]

	$(RUSTC) foo.rs -g
	[ ! -f $(TMPDIR)/*.dwp ]
	[ ! -f $(TMPDIR)/*.dwo ]

packed: packed-split packed-single

packed-split:
	$(RUSTC) foo.rs -g -C split-debuginfo=packed -Z unstable-options -Zsplit-dwarf-kind=split
	ls $(TMPDIR)/*.dwp
	rm -rf $(TMPDIR)/*.dwp $(TMPDIR)/*.dwo

packed-single:
	$(RUSTC) foo.rs -g -C split-debuginfo=packed -Z unstable-options -Zsplit-dwarf-kind=single
	ls $(TMPDIR)/*.dwp
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	rm -rf $(TMPDIR)/*.dwp

unpacked: unpacked-split unpacked-single

unpacked-split:
	$(RUSTC) foo.rs -g -C split-debuginfo=unpacked -Z unstable-options -Zsplit-dwarf-kind=split
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo
	rm -rf $(TMPDIR)/*.dwp $(TMPDIR)/*.dwo

unpacked-single:
	$(RUSTC) foo.rs -g -C split-debuginfo=unpacked -Z unstable-options -Zsplit-dwarf-kind=single
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
endif
endif
