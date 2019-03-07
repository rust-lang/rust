-include ../tools.mk
RUSTC_FLAGS = \
    -l static=bar \
    -l foo \
    -l static=baz \
    -l foo \
    -Z print-link-args

all: $(call DYLIB,foo) $(call STATICLIB,bar) $(call STATICLIB,baz)
	$(RUSTC) $(RUSTC_FLAGS) main.rs
	$(call RUN,main)
