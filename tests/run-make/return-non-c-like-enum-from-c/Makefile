# ignore-cross-compile
include ../tools.mk

all: $(call NATIVE_STATICLIB,test)
	$(RUSTC) nonclike.rs -L$(TMPDIR) -ltest
	$(call RUN,nonclike)
