-include ../tools.mk

all:
	# check that #[cfg_attr(..., ignore)] does the right thing.
	$(RUSTC) --test test-ignore-cfg.rs --cfg ignorecfg
	$(call RUN,test-ignore-cfg) | $(CGREP) 'shouldnotignore - run ... ok' 'shouldignore - ignore ... ignored'
	$(call RUN,test-ignore-cfg --quiet) | $(CGREP) -e "^i\.$$"
	$(call RUN,test-ignore-cfg --quiet) | $(CGREP) -v 'should'
