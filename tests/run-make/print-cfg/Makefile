# needs-llvm-components: x86 arm

include ../tools.mk

all: default output_to_file
	$(RUSTC) --target x86_64-pc-windows-gnu --print cfg | $(CGREP) windows
	$(RUSTC) --target x86_64-pc-windows-gnu --print cfg | $(CGREP) x86_64
	$(RUSTC) --target i686-pc-windows-msvc --print cfg | $(CGREP) msvc
	$(RUSTC) --target i686-apple-darwin --print cfg | $(CGREP) macos
	$(RUSTC) --target i686-unknown-linux-gnu --print cfg | $(CGREP) gnu
	$(RUSTC) --target arm-unknown-linux-gnueabihf --print cfg | $(CGREP) target_abi=
	$(RUSTC) --target arm-unknown-linux-gnueabihf --print cfg | $(CGREP) eabihf

output_to_file:
	# Backend-independent, printed by rustc_driver_impl/src/lib.rs
	-rm $(TMPDIR)/cfg.txt
	$(RUSTC) --target x86_64-pc-windows-gnu --print=cfg -o $(TMPDIR)/cfg.txt
	$(CGREP) windows < $(TMPDIR)/cfg.txt

	# Printed from CodegenBackend trait impl in rustc_codegen_llvm/src/lib.rs
	-rm $(TMPDIR)/cfg.txt
	$(RUSTC) --print=relocation-models -o $(TMPDIR)/cfg.txt
	$(CGREP) dynamic-no-pic < $(TMPDIR)/cfg.txt

	# Printed by compiler/rustc_codegen_llvm/src/llvm_util.rs
	-rm $(TMPDIR)/cfg.txt
	$(RUSTC) --target wasm32-unknown-unknown --print=target-features -o $(TMPDIR)/cfg.txt
	$(CGREP) reference-types < $(TMPDIR)/cfg.txt

	# Printed by C++ code in rustc_llvm/llvm-wrapper/PassWrapper.cpp
	-rm $(TMPDIR)/cfg.txt
	$(RUSTC) --target x86_64-pc-windows-gnu --print=target-cpus -o $(TMPDIR)/cfg.txt
	$(CGREP) native < $(TMPDIR)/cfg.txt

ifdef IS_WINDOWS
default:
	$(RUSTC) --print cfg | $(CGREP) windows
else
default:
	$(RUSTC) --print cfg | $(CGREP) unix
endif
