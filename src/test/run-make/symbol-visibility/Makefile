include ../tools.mk

ifdef IS_WINDOWS
# Do nothing on MSVC.
# On MINGW the --version-script, --dynamic-list, and --retain-symbol args don't
# seem to work reliably.
all:
	exit 0
else

NM=nm -D
DYLIB_EXT=so
CDYLIB_NAME=liba_cdylib.so
RDYLIB_NAME=liba_rust_dylib.so
EXE_NAME=an_executable

ifeq ($(UNAME),Darwin)
NM=nm -gU
DYLIB_EXT=dylib
CDYLIB_NAME=liba_cdylib.dylib
RDYLIB_NAME=liba_rust_dylib.dylib
EXE_NAME=an_executable
endif

all:
	$(RUSTC) an_rlib.rs
	$(RUSTC) a_cdylib.rs
	$(RUSTC) a_rust_dylib.rs
	$(RUSTC) an_executable.rs

	# Check that a cdylib exports its public #[no_mangle] functions
	[ "$$($(NM) $(TMPDIR)/$(CDYLIB_NAME) | grep -c public_c_function_from_cdylib)" -eq "1" ]
	# Check that a cdylib exports the public #[no_mangle] functions of dependencies
	[ "$$($(NM) $(TMPDIR)/$(CDYLIB_NAME) | grep -c public_c_function_from_rlib)" -eq "1" ]
	# Check that a cdylib DOES NOT export any public Rust functions
	[ "$$($(NM) $(TMPDIR)/$(CDYLIB_NAME) | grep -c _ZN.*h.*E)" -eq "0" ]

	# Check that a Rust dylib exports its monomorphic functions
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c public_c_function_from_rust_dylib)" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c _ZN.*public_rust_function_from_rust_dylib.*E)" -eq "1" ]

	# Check that a Rust dylib exports the monomorphic functions from its dependencies
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c public_c_function_from_rlib)" -eq "1" ]
	[ "$$($(NM) $(TMPDIR)/$(RDYLIB_NAME) | grep -c public_rust_function_from_rlib)" -eq "1" ]

	# Check that an executable does not export any dynamic symbols
	[ "$$($(NM) $(TMPDIR)/$(EXE_NAME) | grep -c public_c_function_from_rlib)" -eq "0" ]
	[ "$$($(NM) $(TMPDIR)/$(EXE_NAME) | grep -c public_rust_function_from_exe)" -eq "0" ]

endif
