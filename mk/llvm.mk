# This is just a rough approximation of LLVM deps
LLVM_DEPS:=$(wildcard $(addprefix $(CFG_LLVM_SRC_DIR)/, \
                        * */*h */*/*h */*/*/*h */*cpp */*/*cpp */*/*/*cpp))

define DEF_LLVM_RULES

# If CFG_LLVM_ROOT is defined then we don't build LLVM ourselves
ifeq ($(CFG_LLVM_ROOT),)

$$(LLVM_CONFIG_$(1)): $$(LLVM_DEPS_$(1))
	@$$(call E, make: llvm)
	$$(Q)$$(MAKE) -C $$(CFG_LLVM_BUILD_DIR_$(1))

endif

endef

$(foreach target,$(CFG_TARGET_TRIPLES), \
 $(eval $(call DEF_LLVM_RULES,$(target))))