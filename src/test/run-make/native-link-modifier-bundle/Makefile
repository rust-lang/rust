-include ../../run-make-fulldeps/tools.mk

all: $(call NATIVE_STATICLIB,native-staticlib)
	# Build a staticlib and a rlib, the `native_func` symbol will be bundled into them
	$(RUSTC) bundled.rs --crate-type=staticlib --crate-type=rlib
	"$(LLVM_BIN_DIR)/llvm-nm" $(TMPDIR)/libbundled.a | $(CGREP) -e "T _*native_func"
	"$(LLVM_BIN_DIR)/llvm-nm" $(TMPDIR)/libbundled.a | $(CGREP) -e "U _*native_func"
	"$(LLVM_BIN_DIR)/llvm-nm" $(TMPDIR)/libbundled.rlib | $(CGREP) -e "T _*native_func"
	"$(LLVM_BIN_DIR)/llvm-nm" $(TMPDIR)/libbundled.rlib | $(CGREP) -e "U _*native_func"

	# Build a staticlib and a rlib, the `native_func` symbol will not be bundled into it
	$(RUSTC) non-bundled.rs --crate-type=staticlib --crate-type=rlib
	"$(LLVM_BIN_DIR)/llvm-nm" $(TMPDIR)/libnon_bundled.a | $(CGREP) -ve "T _*native_func"
	"$(LLVM_BIN_DIR)/llvm-nm" $(TMPDIR)/libnon_bundled.a | $(CGREP) -e "U _*native_func"
	"$(LLVM_BIN_DIR)/llvm-nm" $(TMPDIR)/libnon_bundled.rlib | $(CGREP) -ve "T _*native_func"
	"$(LLVM_BIN_DIR)/llvm-nm" $(TMPDIR)/libnon_bundled.rlib | $(CGREP) -e "U _*native_func"

	# Build a cdylib, `native-staticlib` will not appear on the linker line because it was bundled previously
	# The cdylib will contain the `native_func` symbol in the end
	$(RUSTC) cdylib-bundled.rs --crate-type=cdylib --print link-args | $(CGREP) -ve '-l[" ]*native-staticlib|native-staticlib\.lib'
	"$(LLVM_BIN_DIR)/llvm-nm" $(call DYLIB,cdylib_bundled) | $(CGREP) -e "[Tt] _*native_func"

	# Build a cdylib, `native-staticlib` will appear on the linker line because it was not bundled previously
	# The cdylib will contain the `native_func` symbol in the end
	$(RUSTC) cdylib-non-bundled.rs --crate-type=cdylib --print link-args | $(CGREP) -e '-l[" ]*native-staticlib|native-staticlib\.lib'
	"$(LLVM_BIN_DIR)/llvm-nm" $(call DYLIB,cdylib_non_bundled) | $(CGREP) -e "[Tt] _*native_func"
