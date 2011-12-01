# Rules for non-core tools built with the compiler, both for target
# and host architectures

FUZZER_CRATE := $(S)src/fuzzer/fuzzer.rc
FUZZER_INPUTS := $(wildcard $(addprefix $(S)src/fuzzer/, *.rs))

# The test runner that runs the cfail/rfail/rpass and bench tests
COMPILETEST_CRATE := $(S)src/compiletest/compiletest.rc
COMPILETEST_INPUTS := $(wildcard $(S)src/compiletest/*rs)

# Cargo, the package manager
CARGO_CRATE := $(S)src/cargo/cargo.rc
CARGO_INPUTS := $(wildcard $(S)src/cargo/*rs)

# FIXME: These are only built for the host arch. Eventually we'll
# have tools that need to built for other targets.
define TOOLS_STAGE_N

$$(TBIN$(1)_T_$(4)_H_$(3))/fuzzer$$(X):				\
		$$(FUZZER_CRATE) $$(FUZZER_INPUTS)			\
		$$(TSREQ$(1)_T_$(4)_H_$(3))					\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_STDLIB)	\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBRUSTC)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) -o $$@ $$<

# Promote the stageN target to stageN+1 host
# FIXME: Shouldn't need to depend on host/librustc.so once
# rpath is working
$$(HBIN$(2)_H_$(4))/fuzzer$$(X):				\
		$$(TBIN$(1)_T_$(4)_H_$(3))/fuzzer$$(X)	\
		$$(HLIB$(2)_H_$(4))/$$(CFG_LIBRUSTC)	\
		$$(HSREQ$(2)_H_$(4))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TBIN$(1)_T_$(4)_H_$(3))/compiletest$$(X):			\
		$$(COMPILETEST_CRATE) $$(COMPILETEST_INPUTS)	\
		$$(TSREQ$(1)_T_$(4)_H_$(3))						\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_STDLIB)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) -o $$@ $$<

$$(HBIN$(2)_H_$(4))/compiletest$$(X):				\
		$$(TBIN$(1)_T_$(4)_H_$(3))/compiletest$$(X)	\
		$$(HSREQ$(2)_$(4))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TBIN$(1)_T_$(4)_H_$(3))/cargo$$(X):				\
		$$(CARGO_CRATE) $$(CARGO_INPUTS)			\
		$$(TSREQ$(1)_T_$(4)_H_$(3))					\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_STDLIB)   \
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBRUSTC)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) -o $$@ $$<

$$(HBIN$(2)_H_$(4))/cargo$$(X):					\
		$$(TBIN$(1)_T_$(4)_H_$(3))/cargo$$(X)	\
		$$(HSREQ$(2)_$(4))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef

$(foreach host,$(CFG_TARGET_TRIPLES),				\
 $(eval $(call TOOLS_STAGE_N,0,1,$(host),$(host)))	\
 $(eval $(call TOOLS_STAGE_N,1,2,$(host),$(host)))	\
 $(eval $(call TOOLS_STAGE_N,2,3,$(host),$(host))))
