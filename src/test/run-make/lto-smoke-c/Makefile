-include ../tools.mk

ifdef IS_WINDOWS
	EXTRAFLAGS :=
else
ifeq ($(shell uname),Darwin)
else
ifeq ($(shell uname),FreeBSD)
	EXTRAFLAGS := -lm -lpthread -lgcc_s
else
	EXTRAFLAGS := -lm -lrt -ldl -lpthread
endif
endif
endif

# Apparently older versions of GCC segfault if -g is passed...
CC := $(CC:-g=)

all:
	$(RUSTC) foo.rs -Z lto
	ln -s $(call STATICLIB,foo-*) $(call STATICLIB,foo)
	$(CC) bar.c -lfoo -o $(call RUN,bar) $(EXTRAFLAGS) -lstdc++
	$(call RUN,bar)
