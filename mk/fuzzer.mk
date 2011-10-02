FUZZER_CRATE := $(S)src/fuzzer/fuzzer.rc
FUZZER_INPUTS := $(wildcard $(addprefix $(S)src/fuzzer/, *.rs))

define FUZZ_STAGE_N

# We only really care about fuzzing on the host arch
$$(TARGET_BIN$(1)$(CFG_HOST_TRIPLE))/fuzzer$$(X): \
	$$(FUZZER_CRATE) $$(FUZZER_INPUTS) \
	$$(TARGET_SREQ$(1)$(CFG_HOST_TRIPLE)) \
	$$(TARGET_LIB$(1)$(CFG_HOST_TRIPLE))/$$(CFG_STDLIB) \
	$$(TARGET_LIB$(1)$(CFG_HOST_TRIPLE))/$$(CFG_LIBRUSTC)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)) -o $$@ $$<

# Promote the stageN target to stageN+1 host
# FIXME: Shouldn't need to depend on host/librustc.so once
# rpath is working
$$(HOST_BIN$(2))/fuzzer$$(X): \
	$$(TARGET_BIN$(1)$(CFG_HOST_TRIPLE))/fuzzer$$(X) \
	$$(HOST_LIB$(2))/$$(CFG_LIBRUSTC)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef

$(eval $(call FUZZ_STAGE_N,0,1))
$(eval $(call FUZZ_STAGE_N,1,2))
$(eval $(call FUZZ_STAGE_N,2,3))
