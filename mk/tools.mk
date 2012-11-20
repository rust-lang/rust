# Rules for non-core tools built with the compiler, both for target
# and host architectures

FUZZER_LIB := $(S)src/libfuzzer/fuzzer.rc
FUZZER_INPUTS := $(wildcard $(addprefix $(S)src/libfuzzer/, *.rs))

# The test runner that runs the cfail/rfail/rpass and bench tests
COMPILETEST_CRATE := $(S)src/compiletest/compiletest.rc
COMPILETEST_INPUTS := $(wildcard $(S)src/compiletest/*rs)

# Cargo, the package manager
CARGO_LIB := $(S)src/libcargo/cargo.rc
CARGO_INPUTS := $(wildcard $(S)src/libcargo/*rs)

# Rustdoc, the documentation tool
RUSTDOC_LIB := $(S)src/librustdoc/rustdoc.rc
RUSTDOC_INPUTS := $(wildcard $(S)src/librustdoc/*.rs)

# Rusti, the JIT REPL
RUSTI_LIB := $(S)src/librusti/rusti.rc
RUSTI_INPUTS := $(wildcard $(S)src/librusti/*.rs)

# FIXME: These are only built for the host arch. Eventually we'll
# have tools that need to built for other targets.
define TOOLS_STAGE_N_TARGET

$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBFUZZER):          \
		$$(FUZZER_LIB) $$(FUZZER_INPUTS)			\
		$$(TSREQ$(1)_T_$(4)_H_$(3))					\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_CORELIB)	\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_STDLIB)	\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBRUSTC)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) -o $$@ $$< && touch $$@

$$(TBIN$(1)_T_$(4)_H_$(3))/fuzzer$$(X):				\
		$$(DRIVER_CRATE)								\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBFUZZER)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) --cfg fuzzer -o $$@ $$<

$$(TBIN$(1)_T_$(4)_H_$(3))/compiletest$$(X):			\
		$$(COMPILETEST_CRATE) $$(COMPILETEST_INPUTS)	\
		$$(TSREQ$(1)_T_$(4)_H_$(3))						\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_CORELIB)      \
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_STDLIB)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) -o $$@ $$<

$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBCARGO):		\
		$$(CARGO_LIB) $$(CARGO_INPUTS)				\
		$$(TSREQ$(1)_T_$(4)_H_$(3))					\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_CORELIB)	\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_STDLIB)	\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBRUSTC)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) -o $$@ $$< && touch $$@

$$(TBIN$(1)_T_$(4)_H_$(3))/cargo$$(X):				\
		$$(DRIVER_CRATE) 							\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBCARGO)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) --cfg cargo -o $$@ $$<

$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBRUSTDOC):		\
		$$(RUSTDOC_LIB) $$(RUSTDOC_INPUTS)			\
		$$(TSREQ$(1)_T_$(4)_H_$(3))					\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_CORELIB)	\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_STDLIB)	\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBRUSTC)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) -o $$@ $$< && touch $$@

$$(TBIN$(1)_T_$(4)_H_$(3))/rustdoc$$(X):			\
		$$(DRIVER_CRATE) 							\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBRUSTDOC)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) --cfg rustdoc -o $$@ $$<

$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBRUSTI):		\
		$$(RUSTI_LIB) $$(RUSTI_INPUTS)			\
		$$(TSREQ$(1)_T_$(4)_H_$(3))					\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_CORELIB)	\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_STDLIB)	\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBRUSTC)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) -o $$@ $$< && touch $$@

$$(TBIN$(1)_T_$(4)_H_$(3))/rusti$$(X):			\
		$$(DRIVER_CRATE) 							\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBRUSTI)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) --cfg rusti -o $$@ $$<

endef

define TOOLS_STAGE_N_HOST


# Promote the stageN target to stageN+1 host
# FIXME: Shouldn't need to depend on host/librustc.so once
# rpath is working
$$(HLIB$(2)_H_$(4))/$$(CFG_LIBFUZZER):					\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBFUZZER)	\
		$$(HLIB$(2)_H_$(4))/$$(CFG_LIBRUSTC)			\
		$$(HSREQ$(2)_H_$(4))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBFUZZER_GLOB) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBFUZZER_DSYM_GLOB)) \
	        $$(HLIB$(2)_H_$(4))

$$(HBIN$(2)_H_$(4))/fuzzer$$(X):				\
		$$(TBIN$(1)_T_$(4)_H_$(3))/fuzzer$$(X)	\
		$$(HLIB$(2)_H_$(4))/$$(CFG_LIBFUZZER)	\
		$$(HSREQ$(2)_H_$(4))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HBIN$(2)_H_$(4))/compiletest$$(X):				\
		$$(TBIN$(1)_T_$(4)_H_$(3))/compiletest$$(X)	\
		$$(HSREQ$(2)_H_$(4))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@


$$(HLIB$(2)_H_$(4))/$$(CFG_LIBCARGO):				\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBCARGO)	\
		$$(HLIB$(2)_H_$(4))/$$(CFG_LIBRUSTC)		\
		$$(HSREQ$(2)_H_$(4))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBCARGO_GLOB) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBCARGO_DSYM_GLOB)) \
	        $$(HLIB$(2)_H_$(4))

$$(HBIN$(2)_H_$(4))/cargo$$(X):					\
		$$(TBIN$(1)_T_$(4)_H_$(3))/cargo$$(X)	\
		$$(HLIB$(2)_H_$(4))/$$(CFG_LIBCARGO)	\
		$$(HSREQ$(2)_H_$(4))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB$(2)_H_$(4))/$$(CFG_LIBRUSTDOC):					\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBRUSTDOC)	\
		$$(HLIB$(2)_H_$(4))/$$(CFG_LIBRUSTC)			\
		$$(HSREQ$(2)_H_$(4))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTDOC_GLOB) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTDOC_DSYM_GLOB)) \
	        $$(HLIB$(2)_H_$(4))

$$(HBIN$(2)_H_$(4))/rustdoc$$(X):				\
		$$(TBIN$(1)_T_$(4)_H_$(3))/rustdoc$$(X)	\
		$$(HLIB$(2)_H_$(4))/$$(CFG_LIBRUSTDOC)	\
		$$(HSREQ$(2)_H_$(4))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB$(2)_H_$(4))/$$(CFG_LIBRUSTI):					\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBRUSTI)	\
		$$(HLIB$(2)_H_$(4))/$$(CFG_LIBRUSTC)			\
		$$(HSREQ$(2)_H_$(4))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTI_GLOB) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTI_DSYM_GLOB)) \
	        $$(HLIB$(2)_H_$(4))

$$(HBIN$(2)_H_$(4))/rusti$$(X):				\
		$$(TBIN$(1)_T_$(4)_H_$(3))/rusti$$(X)	\
		$$(HLIB$(2)_H_$(4))/$$(CFG_LIBRUSTI)	\
		$$(HSREQ$(2)_H_$(4))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef

$(foreach host,$(CFG_TARGET_TRIPLES),				\
$(foreach target,$(CFG_TARGET_TRIPLES),				\
 $(eval $(call TOOLS_STAGE_N_TARGET,0,1,$(target),$(host)))	\
 $(eval $(call TOOLS_STAGE_N_TARGET,1,2,$(target),$(host)))	\
 $(eval $(call TOOLS_STAGE_N_TARGET,2,3,$(target),$(host)))))

$(foreach host,$(CFG_TARGET_TRIPLES),				\
 $(eval $(call TOOLS_STAGE_N_HOST,0,1,$(host),$(host)))	\
 $(eval $(call TOOLS_STAGE_N_HOST,1,2,$(host),$(host)))	\
 $(eval $(call TOOLS_STAGE_N_HOST,2,3,$(host),$(host))))
