-include ../tools.mk

all:
ifeq ($(PROFILER_SUPPORT),1)
	$(RUSTC) -O -Ccodegen-units=1 -Z pgo-gen="$(TMPDIR)/test.profraw" --emit=llvm-ir test.rs
	# We expect symbols starting with "__llvm_profile_".
	$(CGREP) "__llvm_profile_" < $(TMPDIR)/test.ll
	# We do NOT expect the "__imp_" version of these symbols.
	$(CGREP) -v "__imp___llvm_profile_" < $(TMPDIR)/test.ll # 64 bit
	$(CGREP) -v "__imp____llvm_profile_" < $(TMPDIR)/test.ll # 32 bit
endif
