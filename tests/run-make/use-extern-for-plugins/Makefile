include ../tools.mk

# ignore-freebsd
# ignore-openbsd
# ignore-sunos

HOST := $(shell $(RUSTC) -vV | grep 'host:' | sed 's/host: //')
ifeq ($(findstring i686,$(HOST)),i686)
TARGET := $(subst i686,x86_64,$(HOST))
else
TARGET := $(subst x86_64,i686,$(HOST))
endif

all:
	$(RUSTC) foo.rs -C extra-filename=-host
	$(RUSTC) bar.rs -C extra-filename=-targ --target $(TARGET)
	$(RUSTC) baz.rs --extern a=$(TMPDIR)/liba-targ.rlib --target $(TARGET)
