-include ../tools.mk

all:
	# check that #[ignore(cfg(...))] does the right thing.
	$(RUSTC) --test test-ignore-cfg.rs --cfg ignorecfg
	$(call RUN,test-ignore-cfg) | grep 'shouldnotignore ... ok'
	$(call RUN,test-ignore-cfg) | grep 'shouldignore ... ignored'
