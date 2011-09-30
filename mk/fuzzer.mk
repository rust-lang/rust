FUZZER_CRATE := $(S)src/fuzzer/fuzzer.rc
FUZZER_INPUTS := $(wildcard $(addprefix $(S)src/fuzzer/, *.rs))

define FUZZ_STAGE_N

stage$(2)/bin/fuzzer$$(X): $$(FUZZER_CRATE) $$(FUZZER_INPUTS) \
                          $$(SREQ$(2)$(CFG_HOST_TRIPLE)) \
                          stage$(2)/lib/$$(CFG_RUNTIME)                       \
                          stage$(2)/lib/$$(CFG_RUSTLLVM)                      \
                          stage$(2)/lib/$$(CFG_STDLIB) \
                          stage$(2)/lib/$$(CFG_LIBRUSTC)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)) -L stage1/lib -o $$@ $$<

endef

$(eval $(call FUZZ_STAGE_N,0,1))
$(eval $(call FUZZ_STAGE_N,1,2))
$(eval $(call FUZZ_STAGE_N,2,3))
