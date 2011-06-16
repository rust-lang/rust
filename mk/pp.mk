
PP_INPUTS := $(wildcard $(addprefix $(S)src/lib/,*.rs */*.rs)) \
             $(wildcard $(addprefix $(S)src/comp/,*.rs */*.rs */*/*.rs)) \
             $(wildcard $(S)src/test/*/*.rs   \
                        $(S)src/test/*/*/*.rs)

PP_INPUTS_FILTERED = $(shell echo $(PP_INPUTS) | xargs grep -L no-reformat)

reformat: $(SREQ1)
	@$(call E, reformat [stage1]: $@)
	for i in $(PP_INPUTS_FILTERED);  \
    do $(call CFG_RUN_TARG,stage1, stage1/rustc$(X)) \
       --pretty normal $$i >$$i.tmp; \
    if cmp --silent $$i.tmp $$i; \
        then echo no changes to $$i; rm $$i.tmp; \
        else mv $$i.tmp $$i; fi \
    done
