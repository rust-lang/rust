
ifdef CFG_PERF_TOOL
rustc-perf$(X): stage2/rustc$(X)
	@$(call E, perf compile: $@)
	$(PERF_STAGE1) -L stage2 -o $@ $(COMPILER_CRATE)
	rm -f $@
else
rustc-perf$(X): stage2/rustc$(X)
	touch $@
endif

perf: check-stage2-perf rustc-perf$(X)
