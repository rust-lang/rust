-include ../../run-make-fulldeps/tools.mk

ifdef IS_WINDOWS
all:
	$(RUSTC) -Zunstable-options --print rustc-path | $(CGREP) bin\rustc
else
all:
	$(RUSTC) -Zunstable-options --print rustc-path | $(CGREP) bin/rustc
endif
