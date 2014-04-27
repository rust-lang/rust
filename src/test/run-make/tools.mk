export LD_LIBRARY_PATH:=$(TMPDIR):$(LD_LIBRARY_PATH)
export DYLD_LIBRARY_PATH:=$(TMPDIR):$(DYLD_LIBRARY_PATH)

RUSTC := $(RUSTC) --out-dir $(TMPDIR) -L $(TMPDIR)
CC := $(CC) -L $(TMPDIR)

RUN = $(TMPDIR)/$(1)
FAILS = $(TMPDIR)/$(1) && exit 1 || exit 0

RLIB_GLOB = lib$(1)*.rlib
STATICLIB = $(TMPDIR)/lib$(1).a
STATICLIB_GLOB = lib$(1)*.a
BIN = $(1)

UNAME = $(shell uname)
ifneq (,$(findstring MINGW,$(UNAME)))
IS_WINDOWS=1
endif

ifeq ($(UNAME),Darwin)
DYLIB_GLOB = lib$(1)*.dylib
DYLIB = $(TMPDIR)/lib$(1).dylib
else
ifdef IS_WINDOWS
DYLIB_GLOB = $(1)*.dll
DYLIB = $(TMPDIR)/$(1).dll
BIN = $(1).exe
export PATH := $(PATH):$(LD_LIBRARY_PATH)
else
DYLIB_GLOB = lib$(1)*.so
DYLIB = $(TMPDIR)/lib$(1).so
endif
endif

%.a: %.o
	ar crus $@ $<
%.dylib: %.o
	$(CC) -dynamiclib -Wl,-dylib -o $@ $<
%.so: %.o
	$(CC) -o $@ $< -shared
%.dll: lib%.o
	$(CC) -o $@ $< -shared

$(TMPDIR)/lib%.o: %.c
	$(CC) -c -o $@ $<

