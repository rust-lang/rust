# Recursive wildcard function
# http://blog.jgc.org/2011/07/gnu-make-recursive-wildcard-function.html
rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) \
  $(filter $(subst *,%,$2),$d))

# This is just a rough approximation of LLVM deps
LLVM_DEPS=$(call rwildcard,$(CFG_LLVM_SRC_DIR),*cpp *hpp)

define DEF_LLVM_RULES

# If CFG_LLVM_ROOT is defined then we don't build LLVM ourselves
ifeq ($(CFG_LLVM_ROOT),)

$$(LLVM_CONFIG_$(1)): $$(LLVM_DEPS)
	@$$(call E, make: llvm)
	$$(Q)$$(MAKE) -C $$(CFG_LLVM_BUILD_DIR_$(1)) $$(CFG_LLVM_BUILD_ENV)
	$$(Q)touch $$(LLVM_CONFIG_$(1))
endif

endef

$(foreach target,$(CFG_TARGET_TRIPLES), \
 $(eval $(call DEF_LLVM_RULES,$(target))))