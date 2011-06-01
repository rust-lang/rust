reformat: $(SREQ1)
	@$(call E, reformat [stage1]: $@)
	for i in $(wildcard $(addprefix $(S)src/comp/, \
                          *.rs */*.rs */*/*.rs));  \
    do $(call CFG_RUN_TARG,stage1, stage1/rustc$(X)) \
       --pretty $$i >$$i.tmp && mv $$i.tmp $$i; \
    done
