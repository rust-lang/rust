include ../tools.mk

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
# Windows only supports =packed
off:
packed:
unpacked:
else
ifeq ($(UNAME),Linux)
  UNSTABLEOPTS :=
else
  UNSTABLEOPTS := -Zunstable-options
endif

off:
	$(RUSTC) foo.rs -g -C $(UNSTABLEOPTS) split-debuginfo=off
	[ ! -f $(TMPDIR)/*.dwp ]
	[ ! -f $(TMPDIR)/*.dwo ]

	$(RUSTC) foo.rs -g
	[ ! -f $(TMPDIR)/*.dwp ]
	[ ! -f $(TMPDIR)/*.dwo ]

packed: packed-split packed-single

packed-split:
	$(RUSTC) foo.rs -g $(UNSTABLEOPTS) -C split-debuginfo=packed -Zsplit-dwarf-kind=split
	ls $(TMPDIR)/*.dwp
	rm -rf $(TMPDIR)/*.dwp $(TMPDIR)/*.dwo

packed-single:
	$(RUSTC) foo.rs -g $(UNSTABLEOPTS) -C split-debuginfo=packed -Zsplit-dwarf-kind=single
	ls $(TMPDIR)/*.dwp
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	rm -rf $(TMPDIR)/*.dwp

packed-remapped: packed-remapped-split packed-remapped-single

packed-remapped-split:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=packed -C debuginfo=2 \
		-Z split-dwarf-kind=split --remap-path-prefix $(TMPDIR)=/a foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep DW_AT_GNU_dwo_name | (! grep $(TMPDIR)) || exit 1

packed-remapped-single:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=packed -C debuginfo=2 \
		-Z split-dwarf-kind=single --remap-path-prefix $(TMPDIR)=/a foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep DW_AT_GNU_dwo_name | (! grep $(TMPDIR)) || exit 1

packed-crosscrate: packed-crosscrate-split packed-crosscrate-single

packed-crosscrate-split:
	$(RUSTC) --crate-type lib $(UNSTABLEOPTS) -C split-debuginfo=packed \
		-Zsplit-dwarf-kind=split -C debuginfo=2 -g bar.rs
	ls $(TMPDIR)/*.rlib
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	$(RUSTC) --extern bar=$(TMPDIR)/libbar.rlib -Z unstable-options $(UNSTABLEOPTS) \
		-C split-debuginfo=packed -Zsplit-dwarf-kind=split -C debuginfo=2 -g main.rs
	rm $(TMPDIR)/*.dwo
	rm $(TMPDIR)/main.dwp
	rm $(TMPDIR)/$(call BIN,main)

packed-crosscrate-single:
	$(RUSTC) --crate-type lib $(UNSTABLEOPTS) -C split-debuginfo=packed \
		-Zsplit-dwarf-kind=single -C debuginfo=2 -g bar.rs
	ls $(TMPDIR)/*.rlib
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	$(RUSTC) --extern bar=$(TMPDIR)/libbar.rlib -Z unstable-options $(UNSTABLEOPTS) \
		-C split-debuginfo=packed -Zsplit-dwarf-kind=single -C debuginfo=2 -g main.rs
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	rm $(TMPDIR)/main.dwp
	rm $(TMPDIR)/$(call BIN,main)

unpacked: unpacked-split unpacked-single unpacked-remapped-split unpacked-remapped-single

unpacked-split:
	$(RUSTC) foo.rs -g $(UNSTABLEOPTS) -C split-debuginfo=unpacked -Zsplit-dwarf-kind=split
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo
	rm -rf $(TMPDIR)/*.dwp $(TMPDIR)/*.dwo

unpacked-single:
	$(RUSTC) foo.rs -g $(UNSTABLEOPTS) -C split-debuginfo=unpacked -Zsplit-dwarf-kind=single
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0

unpacked-remapped-split:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		-Z split-dwarf-kind=split --remap-path-prefix $(TMPDIR)=/a foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep DW_AT_GNU_dwo_name | (! grep $(TMPDIR)) || exit 1

unpacked-remapped-single:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		-Z split-dwarf-kind=single --remap-path-prefix $(TMPDIR)=/a foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep DW_AT_GNU_dwo_name | (! grep $(TMPDIR)) || exit 1
endif
endif
