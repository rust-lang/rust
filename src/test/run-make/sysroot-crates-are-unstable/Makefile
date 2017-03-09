-include ../tools.mk

# This is a whitelist of crates which are stable, we don't check for the
# instability of these crates as they're all stable!
STABLE_CRATES := \
	std \
	core \
	proc_macro \
	rsbegin.o \
	rsend.o \
	dllcrt2.o \
	crt2.o

# Generate a list of all crates in the sysroot. To do this we list all files in
# rustc's sysroot, look at the filename, strip everything after the `-`, and
# strip the leading `lib` (if present)
SYSROOT := $(shell $(RUSTC) --print sysroot)
LIBS := $(wildcard $(SYSROOT)/lib/rustlib/$(TARGET)/lib/*)
LIBS := $(foreach lib,$(LIBS),$(notdir $(lib)))
LIBS := $(foreach lib,$(LIBS),$(word 1,$(subst -, ,$(lib))))
LIBS := $(foreach lib,$(LIBS),$(patsubst lib%,%,$(lib)))
LIBS := $(filter-out $(STABLE_CRATES),$(LIBS))

all: $(foreach lib,$(LIBS),check-crate-$(lib)-is-unstable)

check-crate-%-is-unstable:
	@echo verifying $* is an unstable crate
	@echo 'extern crate $*;' | \
		$(RUSTC) - --crate-type rlib 2>&1 | cat > $(TMPDIR)/$*; \
		true
	@grep -q 'use of unstable library feature' $(TMPDIR)/$* || \
		(echo crate $* is not unstable && \
		cat $(TMPDIR)/$* && \
		false)
