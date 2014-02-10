-include ../tools.mk

ifneq ($(shell uname),Darwin)
	EXTRAFLAGS := -lm -lrt -ldl -lpthread
endif

all:
	$(RUSTC) foo.rs -C gen-crate-map
	ln -s $(call STATICLIB,foo-*) $(call STATICLIB,foo)
	$(CC) bar.c -lfoo -o $(call RUN,bar) $(EXTRAFLAGS) -lstdc++
	$(call RUN,bar)
	rm $(call STATICLIB,foo*)
	$(call RUN,bar)
