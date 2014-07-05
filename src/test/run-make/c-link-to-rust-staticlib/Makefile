-include ../tools.mk

ifndef IS_WINDOWS
ifneq ($(shell uname),Darwin)
	EXTRAFLAGS := -lm -lrt -ldl -lpthread
endif
endif

# FIXME: ignore freebsd
ifneq ($(shell uname),FreeBSD)
all:
	$(RUSTC) foo.rs
	$(CC) bar.c -lfoo -o $(call RUN_BINFILE,bar) $(EXTRAFLAGS) -lstdc++
	$(call RUN,bar)
	rm $(call STATICLIB,foo*)
	$(call RUN,bar)

else
all:

endif
