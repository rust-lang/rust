# At the moment the fuzzer only exists in stage2.  That's the first
# stage built by the non-snapshot compiler so it seems a convenient
# stage to work at.

FUZZER_CRATE := $(S)src/fuzzer/fuzzer.rc
FUZZER_INPUTS := $(wildcard $(addprefix $(S)src/fuzzer/, *.rs))

stage2/fuzzer.o: $(FUZZER_CRATE) $(FUZZER_INPUTS) $(SREQ1)
	@$(call E, compile: $@)
	$(STAGE1) -c -o $@ $<
