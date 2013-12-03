-include ../tools.mk

ifneq ($(shell uname),Darwin)
	EXTRAFLAGS := -lm -lrt -ldl -lpthread
endif

all:
	$(RUSTC) foo.rs -Z gen-crate-map -Z lto
	ln -s $(call STATICLIB,foo-*) $(call STATICLIB,foo)
	$(CC) bar.c -lfoo -o $(call RUN,bar) $(EXTRAFLAGS) -lstdc++
	$(call RUN,bar)
