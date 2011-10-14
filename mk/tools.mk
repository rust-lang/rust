# Rules for non-core tools built with the compiler, both for target
# and host architectures

FUZZER_CRATE := $(S)src/fuzzer/fuzzer.rc
FUZZER_INPUTS := $(wildcard $(addprefix $(S)src/fuzzer/, *.rs))

# The test runner that runs the cfail/rfail/rpass and bench tests
COMPILETEST_CRATE := $(S)src/compiletest/compiletest.rc
COMPILETEST_INPUTS := $(wildcard $(S)src/compiletest/*rs)

# FIXME: These are only built for the host arch. Eventually we'll
# have tools that need to built for other targets.
define TOOLS_STAGE_N

$$(TARGET_BIN$(1)$(CFG_HOST_TRIPLE))/fuzzer$$(X): \
	$$(FUZZER_CRATE) $$(FUZZER_INPUTS) \
	$$(TARGET_SREQ$(1)$(CFG_HOST_TRIPLE)) \
	$$(TARGET_LIB$(1)$(CFG_HOST_TRIPLE))/$$(CFG_STDLIB) \
	$$(TARGET_LIB$(1)$(CFG_HOST_TRIPLE))/$$(CFG_LIBRUSTC)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_$(CFG_HOST_TRIPLE)) -o $$@ $$<

# Promote the stageN target to stageN+1 host
# FIXME: Shouldn't need to depend on host/librustc.so once
# rpath is working
$$(HOST_BIN$(2))/fuzzer$$(X): \
	$$(TARGET_BIN$(1)$(CFG_HOST_TRIPLE))/fuzzer$$(X) \
	$$(HOST_LIB$(2))/$$(CFG_LIBRUSTC) \
	$$(HOST_SREQ$(2))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TARGET_BIN$(1)$(CFG_HOST_TRIPLE))/compiletest$$(X): \
	$$(COMPILETEST_CRATE) $$(COMPILETEST_INPUTS) \
	$$(TARGET_SREQ$(1)$(CFG_HOST_TRIPLE)) \
	$$(TARGET_LIB$(1)$(CFG_HOST_TRIPLE))/$$(CFG_STDLIB)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_$(CFG_HOST_TRIPLE)) -o $$@ $$<

$$(HOST_BIN$(2))/compiletest$$(X): \
	$$(TARGET_BIN$(1)$(CFG_HOST_TRIPLE))/compiletest$$(X) \
	$$(HOST_SREQ$(2))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef

$(eval $(call TOOLS_STAGE_N,0,1))
$(eval $(call TOOLS_STAGE_N,1,2))
$(eval $(call TOOLS_STAGE_N,2,3))
