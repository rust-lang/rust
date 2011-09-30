
ifdef CFG_PERF_TOOL
rustc-perf$(X): stage2/bin/rustc$(X)
	@$(call E, perf compile: $@)
	$(PERF_STAGE1) -o $@ $(COMPILER_CRATE) >rustc-perf.err 2>&1
	$(Q)rm -f $@
else
rustc-perf$(X): stage2/bin/rustc$(X)
	$(Q)touch $@
endif

perf: check-stage2-perf rustc-perf$(X)
	$(Q)find test/perf -name \*.err | xargs cat
	$(Q)cat rustc-perf.err
