# Create a way to reformat just some files
ifdef PPFILES
  PP_INPUTS_FILTERED := $(wildcard $(PPFILES))
else
  PP_INPUTS = $(wildcard $(addprefix $(S)src/lib/,*.rs */*.rs)) \
              $(wildcard $(addprefix $(S)src/comp/,*.rs */*.rs */*/*.rs)) \
              $(wildcard $(S)src/test/*/*.rs   \
                         $(S)src/test/*/*/*.rs) \
              $(wildcard $(S)src/fuzzer/*.rs)

  PP_INPUTS_FILTERED = $(shell echo $(PP_INPUTS) | xargs grep -L \
                       "no-reformat\|xfail-pretty\|xfail-stage2")
endif

reformat: $(SREQ1)
	@$(call E, reformat [stage1]: $@)
	for i in $(PP_INPUTS_FILTERED);  \
    do $(call CFG_RUN_TARG,stage1,stage1/rustc$(X)) \
       --pretty normal $$i >$$i.tmp; \
    if [ $$? -ne 0 ]; \
        then echo failed to print $$i; rm $$i.tmp; \
        else if cmp --silent $$i.tmp $$i; \
            then echo no changes to $$i; rm $$i.tmp; \
            else echo reformated $$i; mv $$i.tmp $$i; \
        fi; \
    fi; \
    done
