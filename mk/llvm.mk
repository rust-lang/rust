# This is just a rough approximation of LLVM deps
LLVM_DEPS:=$(wildcard $(addprefix $(CFG_LLVM_SRC_DIR)/, \
                        * */*h */*/*h */*/*/*h */*cpp */*/*cpp */*/*/*cpp))

$(LLVM_CONFIG): $(LLVM_DEPS)
	@$(call E, make: llvm)
	$(Q)make -C $(CFG_LLVM_BUILD_DIR)