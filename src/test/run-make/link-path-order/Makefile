-include ../tools.mk

# Verifies that the -L arguments given to the linker is in the same order
# as the -L arguments on the rustc command line.

CORRECT_DIR=$(TMPDIR)/correct
WRONG_DIR=$(TMPDIR)/wrong

all: $(TMPDIR)/libcorrect.a $(TMPDIR)/libwrong.a
	mkdir -p $(CORRECT_DIR) $(WRONG_DIR)
	mv $(TMPDIR)/libcorrect.a $(CORRECT_DIR)/libfoo.a
	mv $(TMPDIR)/libwrong.a $(WRONG_DIR)/libfoo.a
	$(RUSTC) main.rs -o $(TMPDIR)/should_succeed -L $(CORRECT_DIR) -L $(WRONG_DIR)
	$(call RUN,should_succeed)
	$(RUSTC) main.rs -o $(TMPDIR)/should_fail -L $(WRONG_DIR) -L $(CORRECT_DIR)
	$(call FAIL,should_fail)

