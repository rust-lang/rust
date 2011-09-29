# At the moment the fuzzer only exists in stage1.

FUZZER_CRATE := $(S)src/fuzzer/fuzzer.rc
FUZZER_INPUTS := $(wildcard $(addprefix $(S)src/fuzzer/, *.rs))

stage1/bin/fuzzer$(X): $(FUZZER_CRATE) $(FUZZER_INPUTS) $(SREQ1) \
                   stage1/lib/rustc/$(CFG_HOST_TRIPLE)/$(CFG_LIBRUSTC)
	@$(call E, compile_and_link: $@)
	$(STAGE1) -o $@ $<
