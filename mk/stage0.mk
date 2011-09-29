# FIXME: We're temorarily moving stuff all over the place here to make
# the old snapshot compatible with the new build rules
stage0/bin/rustc$(X): $(S)src/snapshots.txt $(S)src/etc/get-snapshot.py $(MKFILES)
	@$(call E, fetch: $@)
	$(Q)$(S)src/etc/get-snapshot.py
	$(Q)cp stage0/rustc$(X) stage0/bin/rustc$(X)
	$(Q)cp stage0/$(CFG_RUNTIME) stage0/lib/$(CFG_RUNTIME)
	$(Q)cp stage0/$(CFG_RUSTLLVM) stage0/lib/$(CFG_RUSTLLVM)
	$(Q)mkdir -p stage0/bin/lib
	$(Q)cp stage0/lib/intrinsics.bc stage0/bin/lib/intrinsics.bc
	$(Q)cp stage0/lib/glue.o stage0/bin/lib/glue.o
	$(Q)cp stage0/lib/main.o stage0/bin/lib/main.o
	$(Q)touch $@

# Host libs will be made in the process of making rustc above.

stage0/lib/$(CFG_RUNTIME): stage0/bin/rustc$(X)
	$(Q)touch $@

stage0/lib/$(CFG_STDLIB): stage0/bin/rustc$(X)
	$(Q)touch $@

stage0/lib/$(CFG_RUSTLLVM): stage0/bin/rustc$(X)
	$(Q)touch $@

# Instantiate template (in stageN.mk) for building
# target libraries.

SREQpre = stage0/lib/$(CFG_HOST_TRIPLE)/main.o $(MKFILES)
$(eval $(call TARGET_LIBS,pre,0,$(CFG_HOST_TRIPLE)))

