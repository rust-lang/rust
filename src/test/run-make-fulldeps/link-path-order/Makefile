include ../tools.mk

# Verifies that the -L arguments given to the linker is in the same order
# as the -L arguments on the rustc command line.

CORRECT_DIR=$(TMPDIR)/correct
WRONG_DIR=$(TMPDIR)/wrong

F := $(call NATIVE_STATICLIB_FILE,foo)

all: $(call NATIVE_STATICLIB,correct) $(call NATIVE_STATICLIB,wrong)
	mkdir -p $(CORRECT_DIR) $(WRONG_DIR)
	mv $(call NATIVE_STATICLIB,correct) $(CORRECT_DIR)/$(F)
	mv $(call NATIVE_STATICLIB,wrong) $(WRONG_DIR)/$(F)
	$(RUSTC) main.rs -o $(TMPDIR)/should_succeed -L $(CORRECT_DIR) -L $(WRONG_DIR)
	$(call RUN,should_succeed)
	$(RUSTC) main.rs -o $(TMPDIR)/should_fail -L $(WRONG_DIR) -L $(CORRECT_DIR)
	$(call FAIL,should_fail)
