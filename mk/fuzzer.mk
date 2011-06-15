# At the moment the fuzzer only exists in stage2.  That's the first
# stage built by the non-snapshot compiler so it seems a convenient
# stage to work at.

FUZZER_CRATE := $(S)src/fuzzer/fuzzer.rc
FUZZER_INPUTS := $(wildcard $(addprefix $(S)src/fuzzer/, *.rs))

stage2/fuzzer.o: $(FUZZER_CRATE) $(FUZZER_INPUTS) $(SREQ1)
	@$(call E, compile: $@)
	$(STAGE1) -c -o $@ $<

stage2/fuzzer$(X): stage2/fuzzer.o $(SREQ1)
	@$(call E, link [gcc]: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) rt/main.o stage2/glue.o -o $@ $< \
      -Lstage2 -Lrustllvm -Lrt -lrustrt -lrustllvm -lstd -lm -lrustc
	@# dsymutil sometimes fails or prints a warning, but the
	@# program still runs.  Since it simplifies debugging other
	@# programs, I\'ll live with the noise.
	-$(Q)$(CFG_DSYMUTIL) $@
