# ignore-cross-compile
include ../tools.mk

all: off packed unpacked

ifeq ($(UNAME),Darwin)
# If disabled, don't run `dsymutil`.
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
# Windows only supports packed debuginfo - nothing to test.
off:
packed:
unpacked:
else
# Some non-Windows, non-Darwin platforms are not stable, and some are.
ifeq ($(UNAME),Linux)
  UNSTABLEOPTS :=
else
  UNSTABLEOPTS := -Zunstable-options
endif

# - Debuginfo in `.o` files
# - `.o` deleted
# - `.dwo` never created
# - `.dwp` never created
off:
	$(RUSTC) foo.rs -g -C $(UNSTABLEOPTS) split-debuginfo=off
	[ ! -f $(TMPDIR)/*.dwp ]
	[ ! -f $(TMPDIR)/*.dwo ]
	$(RUSTC) foo.rs -g
	[ ! -f $(TMPDIR)/*.dwp ]
	[ ! -f $(TMPDIR)/*.dwo ]

packed: packed-split packed-single packed-lto packed-remapped packed-crosscrate

# - Debuginfo in `.dwo` files
# - `.o` deleted
# - `.dwo` deleted
# - `.dwp` present
packed-split:
	$(RUSTC) foo.rs -g $(UNSTABLEOPTS) -C split-debuginfo=packed -Zsplit-dwarf-kind=split
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	rm $(TMPDIR)/foo.dwp
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.o` files
# - `.o` deleted
# - `.dwo` never created
# - `.dwp` present
packed-single:
	$(RUSTC) foo.rs -g $(UNSTABLEOPTS) -C split-debuginfo=packed -Zsplit-dwarf-kind=single
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	rm $(TMPDIR)/foo.dwp
	rm $(TMPDIR)/$(call BIN,foo)

packed-lto: packed-lto-split packed-lto-single

# - rmeta file added to rlib, no object files are generated and thus no debuginfo is generated
# - `.o` never created
# - `.dwo` never created
# - `.dwp` never created
packed-lto-split:
	$(RUSTC) baz.rs -g $(UNSTABLEOPTS) -Csplit-debuginfo=packed -Zsplit-dwarf-kind=split \
		--crate-type=rlib -Clinker-plugin-lto
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/libbaz.rlib

# - rmeta file added to rlib, no object files are generated and thus no debuginfo is generated
# - `.o` never created
# - `.dwo` never created
# - `.dwp` never created
packed-lto-single:
	$(RUSTC) baz.rs -g $(UNSTABLEOPTS) -Csplit-debuginfo=packed -Zsplit-dwarf-kind=single \
		--crate-type=rlib -Clinker-plugin-lto
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/libbaz.rlib

packed-remapped: packed-remapped-split packed-remapped-single packed-remapped-scope packed-remapped-wrong-scope

# - Debuginfo in `.dwo` files
# - `.o` and binary refer to remapped `.dwo` paths which do not exist
# - `.o` deleted
# - `.dwo` deleted
# - `.dwp` present
packed-remapped-split:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=packed -C debuginfo=2 \
		-Z split-dwarf-kind=split --remap-path-prefix $(TMPDIR)=/a foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep DW_AT_GNU_dwo_name | (! grep $(TMPDIR)) || exit 1
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	rm $(TMPDIR)/foo.dwp
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.o` files
# - `.o` and binary refer to remapped `.o` paths which do not exist
# - `.o` deleted
# - `.dwo` never created
# - `.dwp` present
packed-remapped-single:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=packed -C debuginfo=2 \
		-Z split-dwarf-kind=single --remap-path-prefix $(TMPDIR)=/a foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep DW_AT_GNU_dwo_name | (! grep $(TMPDIR)) || exit 1
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	rm $(TMPDIR)/foo.dwp
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.o` files
# - `.o` and binary refer to remapped `.o` paths which do not exist
# - `.o` deleted
# - `.dwo` never created
# - `.dwp` present
packed-remapped-scope:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=packed -C debuginfo=2 \
		-Z split-dwarf-kind=single --remap-path-prefix $(TMPDIR)=/a \
		-Z remap-path-scope=split-debuginfo-path foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep DW_AT_GNU_dwo_name | (! grep $(TMPDIR)) || exit 1
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	rm $(TMPDIR)/foo.dwp
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.o` files
# - `.o` and binary refer to remapped `.o` paths which do not exist
# - `.o` deleted
# - `.dwo` never created
# - `.dwp` present
packed-remapped-wrong-scope:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=packed -C debuginfo=2 \
		-Z split-dwarf-kind=single --remap-path-prefix $(TMPDIR)=/a \
		-Z remap-path-scope=macro foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep DW_AT_GNU_dwo_name | (grep $(TMPDIR)) || exit 1
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	rm $(TMPDIR)/foo.dwp
	rm $(TMPDIR)/$(call BIN,foo)

packed-crosscrate: packed-crosscrate-split packed-crosscrate-single

# - Debuginfo in `.dwo` files
# - (bar) `.rlib` file created, contains `.dwo`
# - (bar) `.o` deleted
# - (bar) `.dwo` deleted
# - (bar) `.dwp` never created
# - (main) `.o` deleted
# - (main) `.dwo` deleted
# - (main) `.dwp` present
packed-crosscrate-split:
	$(RUSTC) --crate-type lib $(UNSTABLEOPTS) -C split-debuginfo=packed \
		-Zsplit-dwarf-kind=split -C debuginfo=2 -g bar.rs
	ls $(TMPDIR)/*.rlib
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	$(RUSTC) --extern bar=$(TMPDIR)/libbar.rlib $(UNSTABLEOPTS) \
		-C split-debuginfo=packed -Zsplit-dwarf-kind=split -C debuginfo=2 -g main.rs
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	rm $(TMPDIR)/main.dwp
	rm $(TMPDIR)/$(call BIN,main)

# - Debuginfo in `.o` files
# - (bar) `.rlib` file created, contains `.o`
# - (bar) `.o` deleted
# - (bar) `.dwo` never created
# - (bar) `.dwp` never created
# - (main) `.o` deleted
# - (main) `.dwo` never created
# - (main) `.dwp` present
packed-crosscrate-single:
	$(RUSTC) --crate-type lib $(UNSTABLEOPTS) -C split-debuginfo=packed \
		-Zsplit-dwarf-kind=single -C debuginfo=2 -g bar.rs
	ls $(TMPDIR)/*.rlib
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	$(RUSTC) --extern bar=$(TMPDIR)/libbar.rlib $(UNSTABLEOPTS) \
		-C split-debuginfo=packed -Zsplit-dwarf-kind=single -C debuginfo=2 -g main.rs
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	rm $(TMPDIR)/main.dwp
	rm $(TMPDIR)/$(call BIN,main)

unpacked: unpacked-split unpacked-single unpacked-lto unpacked-remapped unpacked-crosscrate

# - Debuginfo in `.dwo` files
# - `.o` deleted
# - `.dwo` present
# - `.dwp` never created
unpacked-split:
	$(RUSTC) foo.rs -g $(UNSTABLEOPTS) -C split-debuginfo=unpacked -Zsplit-dwarf-kind=split
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	rm $(TMPDIR)/*.dwo
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.o` files
# - `.o` present
# - `.dwo` never created
# - `.dwp` never created
unpacked-single:
	$(RUSTC) foo.rs -g $(UNSTABLEOPTS) -C split-debuginfo=unpacked -Zsplit-dwarf-kind=single
	ls $(TMPDIR)/*.o
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/$(call BIN,foo)

unpacked-lto: unpacked-lto-split unpacked-lto-single

# - rmeta file added to rlib, no object files are generated and thus no debuginfo is generated
# - `.o` present (bitcode)
# - `.dwo` never created
# - `.dwp` never created
unpacked-lto-split:
	$(RUSTC) baz.rs -g $(UNSTABLEOPTS) -Csplit-debuginfo=unpacked -Zsplit-dwarf-kind=split \
		--crate-type=rlib -Clinker-plugin-lto
	rm $(TMPDIR)/*.o
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/libbaz.rlib

# - rmeta file added to rlib, no object files are generated and thus no debuginfo is generated
# - `.o` present (bitcode)
# - `.dwo` never created
# - `.dwp` never created
unpacked-lto-single:
	$(RUSTC) baz.rs -g $(UNSTABLEOPTS) -Csplit-debuginfo=unpacked -Zsplit-dwarf-kind=single \
		--crate-type=rlib -Clinker-plugin-lto
	rm $(TMPDIR)/*.o
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/libbaz.rlib

unpacked-remapped: unpacked-remapped-split unpacked-remapped-single unpacked-remapped-scope unpacked-remapped-wrong-scope

# - Debuginfo in `.dwo` files
# - `.o` and binary refer to remapped `.dwo` paths which do not exist
# - `.o` deleted
# - `.dwo` present
# - `.dwp` never created
unpacked-remapped-split:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		-Z split-dwarf-kind=split --remap-path-prefix $(TMPDIR)=/a foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep DW_AT_GNU_dwo_name | (! grep $(TMPDIR)) || exit 1
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	rm $(TMPDIR)/*.dwo
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.o` files
# - `.o` and binary refer to remapped `.o` paths which do not exist
# - `.o` present
# - `.dwo` never created
# - `.dwp` never created
unpacked-remapped-single:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		-Z split-dwarf-kind=single --remap-path-prefix $(TMPDIR)=/a foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep DW_AT_GNU_dwo_name | (! grep $(TMPDIR)) || exit 1
	rm $(TMPDIR)/*.o
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.o` files
# - `.o` and binary refer to remapped `.o` paths which do not exist
# - `.o` present
# - `.dwo` never created
# - `.dwp` never created
unpacked-remapped-scope:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		-Z split-dwarf-kind=single --remap-path-prefix $(TMPDIR)=/a \
		-Z remap-path-scope=split-debuginfo-path foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep DW_AT_GNU_dwo_name | (! grep $(TMPDIR)) || exit 1
	rm $(TMPDIR)/*.o
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.o` files
# - `.o` and binary refer to remapped `.o` paths which do not exist
# - `.o` present
# - `.dwo` never created
# - `.dwp` never created
unpacked-remapped-wrong-scope:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		-Z split-dwarf-kind=single --remap-path-prefix $(TMPDIR)=/a \
		-Z remap-path-scope=macro foo.rs -g
	objdump -Wi $(TMPDIR)/foo | grep DW_AT_GNU_dwo_name | (grep $(TMPDIR)) || exit 1
	rm $(TMPDIR)/*.o
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/$(call BIN,foo)

unpacked-crosscrate: unpacked-crosscrate-split unpacked-crosscrate-single

# - Debuginfo in `.dwo` files
# - (bar) `.rlib` file created, contains `.dwo`
# - (bar) `.o` deleted
# - (bar) `.dwo` present
# - (bar) `.dwp` never created
# - (main) `.o` deleted
# - (main) `.dwo` present
# - (main) `.dwp` never created
unpacked-crosscrate-split:
	$(RUSTC) --crate-type lib $(UNSTABLEOPTS) -C split-debuginfo=unpacked \
		-Zsplit-dwarf-kind=split -C debuginfo=2 -g bar.rs
	ls $(TMPDIR)/*.rlib
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	$(RUSTC) --extern bar=$(TMPDIR)/libbar.rlib $(UNSTABLEOPTS) \
		-C split-debuginfo=unpacked -Zsplit-dwarf-kind=split -C debuginfo=2 -g main.rs
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	rm $(TMPDIR)/*.dwo
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/$(call BIN,main)

# - Debuginfo in `.o` files
# - (bar) `.rlib` file created, contains `.o`
# - (bar) `.o` present
# - (bar) `.dwo` never created
# - (bar) `.dwp` never created
# - (main) `.o` present
# - (main) `.dwo` never created
# - (main) `.dwp` never created
unpacked-crosscrate-single:
	$(RUSTC) --crate-type lib $(UNSTABLEOPTS) -C split-debuginfo=unpacked \
		-Zsplit-dwarf-kind=single -C debuginfo=2 -g bar.rs
	ls $(TMPDIR)/*.rlib
	ls $(TMPDIR)/*.o
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	$(RUSTC) --extern bar=$(TMPDIR)/libbar.rlib $(UNSTABLEOPTS) \
		-C split-debuginfo=unpacked -Zsplit-dwarf-kind=single -C debuginfo=2 -g main.rs
	ls $(TMPDIR)/*.o
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/$(call BIN,main)
endif
endif
