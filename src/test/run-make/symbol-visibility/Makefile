include ../tools.mk

all:
	$(RUSTC) an_rlib.rs
	$(RUSTC) a_cdylib.rs
	$(RUSTC) a_rust_dylib.rs
	$(RUSTC) an_executable.rs

	# Check that a cdylib exports its public #[no_mangle] functions
	[ "$$(nm -D $(TMPDIR)/liba_cdylib.so | grep -c public_c_function_from_cdylib)" -eq "1" ]
	# Check that a cdylib exports the public #[no_mangle] functions of dependencies
	[ "$$(nm -D $(TMPDIR)/liba_cdylib.so | grep -c public_c_function_from_rlib)" -eq "1" ]
	# Check that a cdylib DOES NOT export any public Rust functions
	[ "$$(nm -D $(TMPDIR)/liba_cdylib.so | grep -c _ZN.*h.*E)" -eq "0" ]

	# Check that a Rust dylib exports its monomorphic functions
	[ "$$(nm -D $(TMPDIR)/liba_rust_dylib.so | grep -c public_c_function_from_rust_dylib)" -eq "1" ]
	[ "$$(nm -D $(TMPDIR)/liba_rust_dylib.so | grep -c _ZN.*public_rust_function_from_rust_dylib.*E)" -eq "1" ]

	# Check that a Rust dylib exports the monomorphic functions from its dependencies
	[ "$$(nm -D $(TMPDIR)/liba_rust_dylib.so | grep -c public_c_function_from_rlib)" -eq "1" ]
	[ "$$(nm -D $(TMPDIR)/liba_rust_dylib.so | grep -c public_rust_function_from_rlib)" -eq "1" ]

	# Check that an executable does not export any dynamic symbols
	[ "$$(nm -D $(TMPDIR)/an_executable | grep -c public_c_function_from_rlib)" -eq "0" ]
	[ "$$(nm -D $(TMPDIR)/an_executable | grep -c public_rust_function_from_exe)" -eq "0" ]
