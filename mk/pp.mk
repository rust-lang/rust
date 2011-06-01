
PP_INPUTS := $(wildcard $(addprefix $(S)src/lib/,*.rs */*.rs)) \
             $(wildcard $(addprefix $(S)src/comp/,*.rs */*.rs */*/*.rs))

reformat: $(SREQ1)
	@$(call E, reformat [stage1]: $@)
	for i in $(PP_INPUTS);  \
    do $(call CFG_RUN_TARG,stage1, stage1/rustc$(X)) \
       --pretty $$i >$$i.tmp && mv $$i.tmp $$i; \
    done
