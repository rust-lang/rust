include ../tools.mk

# ignore-windows
#
# On MINGW the --version-script, --dynamic-list, and --retain-symbol args don't
# seem to work reliably.

NM=nm -D
CDYLIB_NAME=liba_cdylib.so
RDYLIB_NAME=liba_rust_dylib.so
EXE_NAME=an_executable
COMBINED_CDYLIB_NAME=libcombined_rlib_dylib.so

ifeq ($(UNAME),Darwin)
NM=nm -gU
CDYLIB_NAME=liba_cdylib.dylib
RDYLIB_NAME=liba_rust_dylib.dylib
EXE_NAME=an_executable
COMBINED_CDYLIB_NAME=libcombined_rlib_dylib.dylib
endif

# `grep` regex for symbols produced by either `legacy` or `v0` mangling
RE_ANY_RUST_SYMBOL="_ZN.*h.*E\|_R[a-zA-Z0-9_]+"

all:
	$(RUSTC) -Zshare-generics=no an_rlib.rs
	$(RUSTC) -Zshare-generics=no a_cdylib.rs
	$(RUSTC) -Zshare-generics=no a_rust_dylib.rs
	$(RUSTC) -Zshare-generics=no an_executable.rs
	$(RUSTC) -Zshare-generics=no a_cdylib.rs --crate-name combined_rlib_dylib --crate-type=rlib,cdylib

	# Check that a cdylib exports its public #[no_mangle] functions
	[ "$$($(NM) $(TMPDIR)/$(CDYLIB_NAME) | grep -c public_c_function_from_cdylib)" -eq "1" ]
	# Check that a cdylib exports the public #[no_mangle] functions of dependencies
	[ "$$($(NM) $(TMPDIR)/$(CDYLIB_NAME) | grep -c public_c_function_from_rlib)" -eq "1" ]
	# Check that a cdylib DOES NOT export any public Rust functions
	[ "$$($(NM) $(TMPDIR)/$(CDYLIB_NAME) | grep -c $(RE_ANY_RUST_SYMBOL))" -eq "0" ]

	# Check that a Rust dylib exports its monomorphic functions
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c public_c_function_from_rust_dylib)" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c public_rust_function_from_rust_dylib)" -eq "1" ]
	# Check that a Rust dylib does not export generics if -Zshare-generics=no
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c public_generic_function_from_rust_dylib)" -eq "0" ]


	# Check that a Rust dylib exports the monomorphic functions from its dependencies
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c public_c_function_from_rlib)" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c public_rust_function_from_rlib)" -eq "1" ]
	# Check that a Rust dylib does not export generics if -Zshare-generics=no
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c public_generic_function_from_rlib)" -eq "0" ]

	# Check that an executable does not export any dynamic symbols
	[ "$$($(NM) $(TMPDIR)/$(EXE_NAME) | grep -c public_c_function_from_rlib)" -eq "0" ]
	[ "$$($(NM) $(TMPDIR)/$(EXE_NAME) | grep -c public_rust_function_from_exe)" -eq "0" ]


	# Check the combined case, where we generate a cdylib and an rlib in the same
	# compilation session:
	# Check that a cdylib exports its public #[no_mangle] functions
	[ "$$($(NM) $(TMPDIR)/$(COMBINED_CDYLIB_NAME) | grep -c public_c_function_from_cdylib)" -eq "1" ]
	# Check that a cdylib exports the public #[no_mangle] functions of dependencies
	[ "$$($(NM) $(TMPDIR)/$(COMBINED_CDYLIB_NAME) | grep -c public_c_function_from_rlib)" -eq "1" ]
	# Check that a cdylib DOES NOT export any public Rust functions
	[ "$$($(NM) $(TMPDIR)/$(COMBINED_CDYLIB_NAME) | grep -c $(RE_ANY_RUST_SYMBOL))" -eq "0" ]


	$(RUSTC) -Zshare-generics=yes an_rlib.rs
	$(RUSTC) -Zshare-generics=yes a_cdylib.rs
	$(RUSTC) -Zshare-generics=yes a_rust_dylib.rs
	$(RUSTC) -Zshare-generics=yes an_executable.rs

	# Check that a cdylib exports its public #[no_mangle] functions
	[ "$$($(NM) $(TMPDIR)/$(CDYLIB_NAME) | grep -c public_c_function_from_cdylib)" -eq "1" ]
	# Check that a cdylib exports the public #[no_mangle] functions of dependencies
	[ "$$($(NM) $(TMPDIR)/$(CDYLIB_NAME) | grep -c public_c_function_from_rlib)" -eq "1" ]
	# Check that a cdylib DOES NOT export any public Rust functions
	[ "$$($(NM) $(TMPDIR)/$(CDYLIB_NAME) | grep -c $(RE_ANY_RUST_SYMBOL))" -eq "0" ]

	# Check that a Rust dylib exports its monomorphic functions, including generics this time
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c public_c_function_from_rust_dylib)" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c public_rust_function_from_rust_dylib)" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c public_generic_function_from_rust_dylib)" -eq "1" ]

	# Check that a Rust dylib exports the monomorphic functions from its dependencies
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c public_c_function_from_rlib)" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c public_rust_function_from_rlib)" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c public_generic_function_from_rlib)" -eq "1" ]

	# Check that an executable does not export any dynamic symbols
	[ "$$($(NM) $(TMPDIR)/$(EXE_NAME) | grep -c public_c_function_from_rlib)" -eq "0" ]
	[ "$$($(NM) $(TMPDIR)/$(EXE_NAME) | grep -c public_rust_function_from_exe)" -eq "0" ]
