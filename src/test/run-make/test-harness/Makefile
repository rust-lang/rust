-include ../tools.mk

all:
	# check that #[cfg_attr(..., ignore)] does the right thing.
	$(RUSTC) --test test-ignore-cfg.rs --cfg ignorecfg
	$(call RUN,test-ignore-cfg) | grep 'shouldnotignore ... ok'
	$(call RUN,test-ignore-cfg) | grep 'shouldignore ... ignored'
	$(call RUN,test-ignore-cfg --quiet) | grep "^i\.$$"
	$(call RUN,test-ignore-cfg --quiet) | grep -v 'should'
