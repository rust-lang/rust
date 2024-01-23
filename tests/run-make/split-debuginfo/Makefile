# ignore-cross-compile
include ../tools.mk

export HERE := $(shell pwd)
export REMAPPED_OUT := /my-remapped-out
export REMAPPED_CWD := /my-remapped-cwd

all: off packed unpacked

ifeq ($(UNAME),Darwin)
# If disabled, don't run `dsymutil`.
off:
	rm -rf $(TMPDIR)/*.dSYM
	$(RUSTC) foo.rs -g -C split-debuginfo=off
	[ ! -d $(TMPDIR)/foo.dSYM ]

# Packed by default, but only if debuginfo is requested
packed: packed-remapped-object-scope packed-remapped-macro-scope packed-remapped-split-debuginfo-scope

# - Debuginfo in binary file
# - `.o` deleted
# - `.dSYM` present
packed-remapped-object-scope:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=packed -C debuginfo=2 \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		-Z remap-path-scope=object foo.rs -g
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	[ -d $(TMPDIR)/foo.dSYM ]

	# binary contains remapped paths
	# FIXME: As of 2023-12, `OSO` cannot be trimmed. See rust-lang/rust#116948
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo) || exit 1
	# FIXME: `SO` in the root debuginfo compile unit didn't get trimed. See rust-lang/rust#116948
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo) || exit 1

	# binary contains no original paths
	# FIXME: As of 2023-12, `OSO` cannot be trimmed. See rust-lang/rust#116948
	grep --text $(TMPDIR) $(TMPDIR)/foo || exit 1
	# FIXME: `SO` in the root debuginfo compile unit didn't get trimed. See rust-lang/rust#116948
	grep --text $(HERE) $(TMPDIR)/foo || exit 1

	# dSYM bundle contains no remapped paths
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo.dSYM/Contents/Resources/DWARF/foo) || exit 1
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo.dSYM/Contents/Resources/DWARF/foo) || exit 1

	# dSYM bundle contains original paths
	grep --text $(HERE) $(TMPDIR)/foo.dSYM/Contents/Resources/DWARF/foo || exit 1

	rm -rf $(TMPDIR)/foo.dSYM
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in binary file
# - `.o` deleted
# - `.dSYM` present
packed-remapped-macro-scope:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=packed -C debuginfo=2 \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		-Z remap-path-scope=macro foo.rs -g
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	[ -d $(TMPDIR)/foo.dSYM ]

	# binary contains no remapped paths
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo) || exit 1
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo) || exit 1

	# binary contains original paths
	grep --text $(TMPDIR) $(TMPDIR)/foo || exit 1
	grep --text $(HERE) $(TMPDIR)/foo || exit 1

	# dSYM bundle contains no remapped paths
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo.dSYM/Contents/Resources/DWARF/foo) || exit 1
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo.dSYM/Contents/Resources/DWARF/foo) || exit 1

	# dSYM bundle contains original paths
	grep --text $(HERE) $(TMPDIR)/foo.dSYM/Contents/Resources/DWARF/foo || exit 1

	rm -rf $(TMPDIR)/foo.dSYM
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in binary file
# - `.o` deleted
# - `.dSYM` present
packed-remapped-split-debuginfo-scope:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=packed -C debuginfo=2 \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		-Z remap-path-scope=split-debuginfo,split-debuginfo-path foo.rs -g
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	[ -d $(TMPDIR)/foo.dSYM ]

	# binary contains no remapped paths
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo) || exit 1
	# FIXME: binary contains `N_SO` (SO) in debug symbol, which got remapped.
	nm -pa $(TMPDIR)/foo | grep " SO " | grep $(REMAPPED_CWD) || exit 1
	nm -pa $(TMPDIR)/foo | grep --invert-match " SO " | (! grep $(REMAPPED_CWD)) || exit 1

	# binary contains original paths
	# FIXME: As of 2023-12, `OSO` cannot be trimmed. See rust-lang/rust#116948
	grep --text $(TMPDIR) $(TMPDIR)/foo || exit 1
	# FIXME: binary contains `N_SO` (SO) in debug symbol, which got remapped.
	(! grep --text $(HERE) $(TMPDIR)/foo) || exit 1

	## dSYM bundle contains remapped paths
	grep --text $(REMAPPED_CWD) $(TMPDIR)/foo.dSYM/Contents/Resources/DWARF/foo || exit 1

	## dSYM bundle contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo.dSYM/Contents/Resources/DWARF/foo) || exit 1
	(! grep --text $(HERE) $(TMPDIR)/foo.dSYM/Contents/Resources/DWARF/foo) || exit 1

	rm -rf $(TMPDIR)/foo.dSYM
	rm $(TMPDIR)/$(call BIN,foo)

# Object files are preserved with unpacked and `dsymutil` isn't run
unpacked: unpacked-remapped-object-scope unpacked-remapped-macro-scope unpacked-remapped-split-debuginfo-scope

# - Debuginfo in object files
# - `.o` present
# - `.dSYM` never created
unpacked-remapped-object-scope:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		-Z remap-path-scope=object foo.rs -g
	ls $(TMPDIR)/*.o
	[ ! -d $(TMPDIR)/foo.dSYM ]

	# binary contains remapped paths
	# FIXME: As of 2023-12, `OSO` cannot be trimmed. See rust-lang/rust#116948
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo) || exit 1
	# FIXME: `SO` in the root debuginfo compile unit didn't get trimed. See rust-lang/rust#116948
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo) || exit 1

	# binary contains no original paths
	# FIXME: As of 2023-12, `OSO` cannot be trimmed. See rust-lang/rust#116948
	grep --text $(TMPDIR) $(TMPDIR)/foo || exit 1
	# FIXME: `SO` in the root debuginfo compile unit didn't get trimed. See rust-lang/rust#116948
	grep --text $(HERE) $(TMPDIR)/foo || exit 1

	# object file contains no remapped paths
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo.*.o) || exit 1
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo.*.o) || exit 1

	# object file contains original paths
	grep --text $(HERE) $(TMPDIR)/foo.*.o || exit 1

	rm $(TMPDIR)/*.o
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in object files
# - `.o` present,
# - `.dSYM` never created
unpacked-remapped-macro-scope:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		-Z remap-path-scope=macro foo.rs -g
	ls $(TMPDIR)/*.o
	[ ! -d $(TMPDIR)/foo.dSYM ]

	# binary contains no remapped paths
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo) || exit 1
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo) || exit 1

	# binary contains original paths
	grep --text $(TMPDIR) $(TMPDIR)/foo || exit 1
	grep --text $(HERE) $(TMPDIR)/foo || exit 1

	# object file contains no remapped paths
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo.*.o) || exit 1
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo.*.o) || exit 1

	# object file contains original paths
	grep --text $(HERE) $(TMPDIR)/foo.*.o || exit 1

	rm $(TMPDIR)/*.o
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in object files
# - `.o` present
# - `.dSYM` never created
unpacked-remapped-split-debuginfo-scope:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		-Z remap-path-scope=split-debuginfo,split-debuginfo-path foo.rs -g
	ls $(TMPDIR)/*.o
	[ ! -d $(TMPDIR)/foo.dSYM ]

	# binary contains no remapped paths
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo) || exit 1
	# FIXME: binary contains `N_SO` (SO) in debug symbol, which got remapped.
	nm -pa $(TMPDIR)/foo | grep " SO " | grep $(REMAPPED_CWD) || exit 1
	nm -pa $(TMPDIR)/foo | grep --invert-match " SO " | (! grep $(REMAPPED_CWD)) || exit 1

	# binary contains original paths
	grep --text $(TMPDIR) $(TMPDIR)/foo || exit 1
	# FIXME: binary contains `N_SO` (SO) in debug symbol, which got remapped.
	(! grep --text $(HERE) $(TMPDIR)/foo) || exit 1

	# object file contains remapped paths
	grep --text $(REMAPPED_CWD) $(TMPDIR)/foo.*.o || exit 1

	# object file contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo.*.o) || exit 1
	(! grep --text $(HERE) $(TMPDIR)/foo.*.o) || exit 1

	rm $(TMPDIR)/*.o
	rm $(TMPDIR)/$(call BIN,foo)

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

packed-remapped: packed-remapped-split packed-remapped-single \
	packed-remapped-object-scope packed-remapped-macro-scope \
	packed-remapped-split-debuginfo-scope

# - Debuginfo in `.dwo` files
# - `.o` and binary refer to remapped `.dwo` paths which do not exist
# - `.o` deleted
# - `.dwo` deleted
# - `.dwp` present
packed-remapped-split:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=packed -C debuginfo=2 \
		-Z split-dwarf-kind=split \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		foo.rs -g

	# binary contains remapped paths
	grep --text $(REMAPPED_OUT) $(TMPDIR)/foo || exit 1
	grep --text $(REMAPPED_CWD) $(TMPDIR)/foo || exit 1

	# binary contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo) || exit 1
	(! grep --text $(HERE) $(TMPDIR)/foo) || exit 1

	# dwp contains remapped paths
	grep --text $(REMAPPED_OUT) $(TMPDIR)/foo.dwp || exit 1

	# dwp contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo.dwp) || exit 1
	(! grep --text $(HERE) $(TMPDIR)/foo.dwp) || exit 1

	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	rm $(TMPDIR)/foo.dwp
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.dwp` files
# - `.o` and binary refer to remapped `.o` paths which do not exist
# - `.o` deleted
# - `.dwo` never created
# - `.dwp` present
packed-remapped-single:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=packed -C debuginfo=2 \
		-Z split-dwarf-kind=single \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		foo.rs -g

	# binary contains remapped paths
	grep --text $(REMAPPED_OUT) $(TMPDIR)/foo || exit 1
	grep --text $(REMAPPED_CWD) $(TMPDIR)/foo || exit 1

	# binary contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo) || exit 1
	(! grep --text $(HERE) $(TMPDIR)/foo) || exit 1

	# dwp contains remapped paths
	grep --text $(REMAPPED_OUT) $(TMPDIR)/foo.dwp || exit 1

	# dwp contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo.dwp) || exit 1
	(! grep --text $(HERE) $(TMPDIR)/foo.dwp) || exit 1

	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	rm $(TMPDIR)/foo.dwp
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.dwp` files
# - `.o` and binary refer to remapped `.o` paths which do not exist
# - `.o` deleted
# - `.dwo` never created
# - `.dwp` present
packed-remapped-object-scope:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=packed -C debuginfo=2 \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		-Z remap-path-scope=object foo.rs -g

	# binary contains remapped paths
	grep --text $(REMAPPED_OUT) $(TMPDIR)/foo || exit 1
	# FIXME: binary contains `DW_AT_comp_dir` in debug symbol, which should be remapped.
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo) || exit 1

	# binary contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo) || exit 1
	# FIXME: bnary contains `DW_AT_comp_dir` in debug symbol, which should be remapped.
	readelf -wi $(TMPDIR)/foo | grep "DW_AT_comp_dir" | grep $(HERE) || exit 1
	readelf -wi $(TMPDIR)/foo | grep --invert-match "DW_AT_comp_dir" | (! grep $(HERE)) || exit 1

	# dwp contains no remapped paths
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo.dwp) || exit 1
	# FIXME: dwp contains `DW_AT_GNU_dwo_name` in debug symbol, which got remapped.
	readelf -wi $(TMPDIR)/foo.dwp | grep "DW_AT_GNU_dwo_name" | grep $(REMAPPED_OUT) || exit 1
	readelf -wi $(TMPDIR)/foo.dwp | grep --invert-match "DW_AT_GNU_dwo_name" | (! grep $(REMAPPED_OUT)) || exit 1

	# dwp contains original paths
	# FIXME: dwp contains `DW_AT_GNU_dwo_name` in debug symbol, which got remapped.
	(! grep --text $(TMPDIR) $(TMPDIR)/foo.dwp) || exit 1

	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	rm $(TMPDIR)/foo.dwp
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.dwp` files
# - `.o` and binary refer to remapped `.o` paths which do not exist
# - `.o` deleted
# - `.dwo` never created
# - `.dwp` present
packed-remapped-macro-scope:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=packed -C debuginfo=2 \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		-Z remap-path-scope=macro foo.rs -g

	# binary contains no remapped paths
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo) || exit 1
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo) || exit 1

	# binary contains original paths
	grep --text $(TMPDIR) $(TMPDIR)/foo || exit 1
	grep --text $(HERE) $(TMPDIR)/foo || exit 1

	# dwp contains no remapped paths
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo.dwp) || exit 1
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo.dwp) || exit 1

	# dwp contains original paths
	grep --text $(TMPDIR) $(TMPDIR)/foo.dwp || exit 1

	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	rm $(TMPDIR)/foo.dwp
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.dwp` files
# - `.o` deleted
# - `.dwo` never created
# - `.dwp` present
packed-remapped-split-debuginfo-scope:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=packed -C debuginfo=2 \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		-Z remap-path-scope=split-debuginfo,split-debuginfo-path foo.rs -g

	# binary contains remapped dwo path
	readelf -wi $(TMPDIR)/foo | grep "DW_AT_GNU_dwo_name" | grep $(REMAPPED_OUT) || exit 1
	readelf -wi $(TMPDIR)/foo | grep --invert-match "DW_AT_GNU_dwo_name" | (! grep $(REMAPPED_OUT)) || exit 1
	# FIXME: binary contains `DW_AT_comp_dir` in debug symbol, which got remapped.
	readelf -wi $(TMPDIR)/foo | grep "DW_AT_comp_dir" | grep $(REMAPPED_CWD) || exit 1
	readelf -wi $(TMPDIR)/foo | grep --invert-match "DW_AT_comp_dir" | (! grep $(REMAPPED_CWD)) || exit 1

	# binary contains no original --out-dir (DW_AT_GNU_dwo_name was remapped)
	(! grep --text $(TMPDIR) $(TMPDIR)/foo) || exit 1
	# FIXME: binary contains `DW_AT_comp_dir` in debug symbol, which got remapped.
	(! grep --text $(HERE) $(TMPDIR)/foo) || exit 1

	# dwp contains remapped paths
	grep --text $(REMAPPED_OUT) $(TMPDIR)/foo.dwp || exit 1

	# dwp contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo.dwp) || exit 1
	(! grep --text $(HERE) $(TMPDIR)/foo.dwp) || exit 1

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

unpacked-lto: packed-lto-split packed-lto-single

# - rmeta file added to rlib, no object files are generated and thus no debuginfo is generated
# - `.o` never created
# - `.dwo` never created
# - `.dwp` never created
unpacked-lto-split:
	$(RUSTC) baz.rs -g $(UNSTABLEOPTS) -Csplit-debuginfo=unpacked -Zsplit-dwarf-kind=split \
		--crate-type=rlib -Clinker-plugin-lto
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/libbaz.rlib

# - rmeta file added to rlib, no object files are generated and thus no debuginfo is generated
# - `.o` never created
# - `.dwo` never created
# - `.dwp` never created
unpacked-lto-single:
	$(RUSTC) baz.rs -g $(UNSTABLEOPTS) -Csplit-debuginfo=unpacked -Zsplit-dwarf-kind=single \
		--crate-type=rlib -Clinker-plugin-lto
	ls $(TMPDIR)/*.o && exit 1 || exit 0
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/libbaz.rlib

unpacked-remapped: unpacked-remapped-split unpacked-remapped-single \
	unpacked-remapped-object-scope unpacked-remapped-macro-scope \
	unpacked-remapped-split-debuginfo-scope

# - Debuginfo in `.dwo` files
# - `.o` deleted
# - `.dwo` present
# - `.dwp` never created
unpacked-remapped-split:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		-Z split-dwarf-kind=split \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		foo.rs -g

	# binary contains remapped paths
	grep --text $(REMAPPED_OUT) $(TMPDIR)/foo || exit 1
	grep --text $(REMAPPED_CWD) $(TMPDIR)/foo || exit 1

	# binary contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo) || exit 1
	(! grep --text $(HERE) $(TMPDIR)/foo) || exit 1

	# dwo contains remapped paths
	grep --text $(REMAPPED_OUT) $(TMPDIR)/foo.*.dwo || exit 1

	# dwo contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo.*.dwo) || exit 1
	(! grep --text $(HERE) $(TMPDIR)/foo.*.dwo) || exit 1

	ls $(TMPDIR)/*.o && exit 1 || exit 0
	rm $(TMPDIR)/*.dwo
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.o` files
# - `.o` present
# - `.dwo` never created
# - `.dwp` never created
unpacked-remapped-single:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		-Z split-dwarf-kind=single \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		foo.rs -g

	# binary contains remapped paths
	grep --text $(REMAPPED_OUT) $(TMPDIR)/foo || exit 1
	grep --text $(REMAPPED_CWD) $(TMPDIR)/foo || exit 1

	# binary contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo) || exit 1
	(! grep --text $(HERE) $(TMPDIR)/foo) || exit 1

	# object file contains remapped paths
	grep --text $(REMAPPED_OUT) $(TMPDIR)/foo.*.o || exit 1
	grep --text $(REMAPPED_CWD) $(TMPDIR)/foo.*.o || exit 1

	# object file contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo.*.o) || exit 1
	(! grep --text $(HERE) $(TMPDIR)/foo.*.o) || exit 1

	rm $(TMPDIR)/*.o
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/$(call BIN,foo)

unpacked-remapped-object-scope: unpacked-remapped-object-scope-split unpacked-remapped-object-scope-single

# - Debuginfo in `.dwo` files
# - `.o` and binary refer to remapped `.dwo` paths which do not exist
# - `.o` deleted
# - `.dwo` present
# - `.dwp` never created
unpacked-remapped-object-scope-split:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		-Z split-dwarf-kind=split \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		-Z remap-path-scope=object foo.rs -g

	# binary contains remapped paths
	grep --text $(REMAPPED_OUT) $(TMPDIR)/foo || exit 1
	# FIXME: binary should contains no cwd but now it got `DW_AT_comp_dir`
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo) || exit 1

	# binary contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo) || exit 1
	# FIXME: binary contains `DW_AT_comp_dir` in debug symbol, which didn't get remapped.
	readelf -wi $(TMPDIR)/foo | grep "DW_AT_comp_dir" | grep $(HERE) || exit 1
	readelf -wi $(TMPDIR)/foo | grep --invert-match "DW_AT_comp_dir" | (! grep $(HERE)) || exit 1

	# dwo contains no remapped paths
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo.*.dwo) || exit 1
	# FIXME: dwo contains `DW_AT_GNU_dwo_name` in debug symbol, which got remapped.
	readelf -wi $(TMPDIR)/foo.*.dwo | grep "DW_AT_GNU_dwo_name" | grep $(REMAPPED_OUT) || exit 1
	readelf -wi $(TMPDIR)/foo.*.dwo | grep --invert-match "DW_AT_GNU_dwo_name" | (! grep $(REMAPPED_OUT)) || exit 1

	# dwo contains original paths
	# FIXME: dwo contains `DW_AT_GNU_dwo_name` in debug symbol, which got remapped.
	(! grep --text $(TMPDIR) $(TMPDIR)/foo.*.dwo) || exit 1

	ls $(TMPDIR)/*.o && exit 1 || exit 0
	rm $(TMPDIR)/*.dwo
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.o` files
# - `.o` and binary refer to remapped `.o` paths which do not exist
# - `.o` present
# - `.dwo` never created
# - `.dwp` never created
unpacked-remapped-object-scope-single:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		-Z split-dwarf-kind=single \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		-Z remap-path-scope=object foo.rs -g

	# binary contains remapped paths
	grep --text $(REMAPPED_OUT) $(TMPDIR)/foo || exit 1
	# FIXME: binary should contains no cwd but now it got `DW_AT_comp_dir`
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo) || exit 1

	# binary contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo) || exit 1
	# FIXME: binary contains `DW_AT_comp_dir` in debug symbol, which didn't get remapped.
	readelf -wi $(TMPDIR)/foo | grep "DW_AT_comp_dir" | grep $(HERE) || exit 1
	readelf -wi $(TMPDIR)/foo | grep --invert-match "DW_AT_comp_dir" | (! grep $(HERE)) || exit 1

	# object file contains no remapped paths
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo.*.o) || exit 1
	# FIXME: object file contains `DW_AT_GNU_dwo_name` in debug symbol, which got remapped.
	readelf -wi $(TMPDIR)/foo.*.o | grep "DW_AT_GNU_dwo_name" | grep $(REMAPPED_OUT) || exit 1
	readelf -wi $(TMPDIR)/foo.*.o | grep --invert-match "DW_AT_GNU_dwo_name" | (! grep $(REMAPPED_OUT)) || exit 1

	# object file contains original paths
	grep --text $(HERE) $(TMPDIR)/foo.*.o || exit 1
	# FIXME: object file contains `DW_AT_GNU_dwo_name` in debug symbol, which got remapped.
	(! grep --text $(TMPDIR) $(TMPDIR)/foo.*.o) || exit 1

	rm $(TMPDIR)/*.o
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/$(call BIN,foo)

unpacked-remapped-macro-scope: unpacked-remapped-macro-scope-split unpacked-remapped-macro-scope-single

# - Debuginfo in `.dwo` files
# - `.o` deleted
# - `.dwo` present
# - `.dwp` never created
unpacked-remapped-macro-scope-split:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		-Z split-dwarf-kind=split \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		-Z remap-path-scope=macro foo.rs -g

	# binary contains no remapped paths
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo) || exit 1
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo) || exit 1

	# binary contains original paths
	grep --text $(TMPDIR) $(TMPDIR)/foo || exit 1
	grep --text $(HERE) $(TMPDIR)/foo || exit 1

	# dwo contains no remapped paths
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo.*.dwo) || exit 1
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo.*.dwo) || exit 1

	# dwo contains original paths
	grep --text $(TMPDIR) $(TMPDIR)/foo.*.dwo || exit 1

	ls $(TMPDIR)/*.o && exit 1 || exit 0
	rm $(TMPDIR)/*.dwo
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.o` files
# - `.o` present
# - `.dwo` never created
# - `.dwp` never created
unpacked-remapped-macro-scope-single:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		-Z split-dwarf-kind=single \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		-Z remap-path-scope=macro foo.rs -g

	# binary contains no remapped paths
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo) || exit 1
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo) || exit 1

	# binary contains original paths
	grep --text $(TMPDIR) $(TMPDIR)/foo || exit 1
	grep --text $(HERE) $(TMPDIR)/foo || exit 1

	# object file contains no remapped paths
	(! grep --text $(REMAPPED_OUT) $(TMPDIR)/foo.*.o) || exit 1
	(! grep --text $(REMAPPED_CWD) $(TMPDIR)/foo.*.o) || exit 1

	# object file contains original paths
	grep --text $(TMPDIR) $(TMPDIR)/foo.*.o || exit 1
	grep --text $(HERE) $(TMPDIR)/foo.*.o || exit 1

	rm $(TMPDIR)/*.o
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/*.dwp && exit 1 || exit 0
	rm $(TMPDIR)/$(call BIN,foo)

unpacked-remapped-split-debuginfo-scope: unpacked-remapped-split-debuginfo-scope-split \
	unpacked-remapped-split-debuginfo-scope-single

# - Debuginfo in `.dwo` files
# - `.o` deleted
# - `.dwo` present
# - `.dwp` never created
unpacked-remapped-split-debuginfo-scope-split:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		-Z split-dwarf-kind=split \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		-Z remap-path-scope=split-debuginfo,split-debuginfo-path foo.rs -g

	# binary contains remapped dwo path
	readelf -wi $(TMPDIR)/foo | grep "DW_AT_GNU_dwo_name" | grep $(REMAPPED_OUT) || exit 1
	readelf -wi $(TMPDIR)/foo | grep --invert-match "DW_AT_GNU_dwo_name" | (! grep $(REMAPPED_OUT)) || exit 1
	# FIXME: binary contains `DW_AT_comp_dir` in debug symbol, which got remapped.
	readelf -wi $(TMPDIR)/foo | grep "DW_AT_comp_dir" | grep $(REMAPPED_CWD) || exit 1
	readelf -wi $(TMPDIR)/foo | grep --invert-match "DW_AT_comp_dir" | (! grep $(REMAPPED_CWD)) || exit 1

	# binary contains no original --out-dir (DW_AT_GNU_dwo_name was remapped)
	(! grep --text $(TMPDIR) $(TMPDIR)/foo) || exit 1
	# FIXME: binary contains `DW_AT_comp_dir` in debug symbol, which got remapped.
	(! grep --text $(HERE) $(TMPDIR)/foo) || exit 1

	# dwo contains remapped paths
	grep --text $(REMAPPED_OUT) $(TMPDIR)/foo.*.dwo || exit 1

	# dwo contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo.*.dwo) || exit 1
	(! grep --text $(HERE) $(TMPDIR)/foo.*.dwo) || exit 1

	ls $(TMPDIR)/*.o && exit 1 || exit 0
	rm $(TMPDIR)/*.dwo
	ls $(TMPDIR)/foo.dwp && exit 1 || exit 0
	rm $(TMPDIR)/$(call BIN,foo)

# - Debuginfo in `.o` files
# - `.o` present
# - `.dwo` never created
# - `.dwp` never created
unpacked-remapped-split-debuginfo-scope-single:
	$(RUSTC) $(UNSTABLEOPTS) -C split-debuginfo=unpacked -C debuginfo=2 \
		-Z split-dwarf-kind=single \
		--remap-path-prefix $(TMPDIR)=$(REMAPPED_OUT) \
		--remap-path-prefix $(HERE)=$(REMAPPED_CWD) \
		-Z remap-path-scope=split-debuginfo,split-debuginfo-path foo.rs -g

	# binary contains remapped dwo path
	readelf -wi $(TMPDIR)/foo | grep "DW_AT_GNU_dwo_name" | grep $(REMAPPED_OUT) || exit 1
	readelf -wi $(TMPDIR)/foo | grep --invert-match "DW_AT_GNU_dwo_name" | (! grep $(REMAPPED_OUT)) || exit 1
	# FIXME: binary contains `DW_AT_comp_dir` in debug symbol, which got remapped.
	readelf -wi $(TMPDIR)/foo | grep "DW_AT_comp_dir" | grep $(REMAPPED_CWD) || exit 1
	readelf -wi $(TMPDIR)/foo | grep --invert-match "DW_AT_comp_dir" | (! grep $(REMAPPED_CWD)) || exit 1

	# binary contains no original --out-dir (DW_AT_GNU_dwo_name was remapped)
	(! grep --text $(TMPDIR) $(TMPDIR)/foo) || exit 1
	# FIXME: binary contains `DW_AT_comp_dir` in debug symbol, which got remapped.
	(! grep --text $(HERE) $(TMPDIR)/foo) || exit 1

	# object file contains remapped paths
	grep --text $(REMAPPED_OUT) $(TMPDIR)/foo.*.o || exit 1

	# object file contains no original paths
	(! grep --text $(TMPDIR) $(TMPDIR)/foo.*.o) || exit 1
	(! grep --text $(HERE) $(TMPDIR)/foo.*.o) || exit 1

	rm $(TMPDIR)/*.o
	ls $(TMPDIR)/*.dwo && exit 1 || exit 0
	ls $(TMPDIR)/foo.dwp && exit 1 || exit 0
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
