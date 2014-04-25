export LD_LIBRARY_PATH:=$(TMPDIR):$(LD_LIBRARY_PATH)
export DYLD_LIBRARY_PATH:=$(TMPDIR):$(DYLD_LIBRARY_PATH)

RUSTC := $(RUSTC) --out-dir $(TMPDIR) -L $(TMPDIR)
CC := $(CC) -L $(TMPDIR)

# This is the name of the binary we will generate and run; use this
# e.g. for `$(CC) -o $(RUN_BINFILE)`.
RUN_BINFILE = $(TMPDIR)/$(1)
# This the basic way we will invoke the generated binary.  It sets the
# LD_LIBRARY_PATH environment variable before running the binary.
RUN = $(TARGET_RPATH_ENV) $(RUN_BINFILE)
FAILS = $(TARGET_RPATH_ENV) ( $(RUN_BINFILE) && exit 1 || exit 0 )

RLIB_GLOB = lib$(1)*.rlib
STATICLIB = $(TMPDIR)/lib$(1).a
STATICLIB_GLOB = lib$(1)*.a

ifeq ($(shell uname),Darwin)
DYLIB_GLOB = lib$(1)*.dylib
DYLIB = $(TMPDIR)/lib$(1).dylib
else
DYLIB_GLOB = lib$(1)*.so
DYLIB = $(TMPDIR)/lib$(1).so
endif

%.a: %.o
	ar crus $@ $<
%.dylib: %.o
	$(CC) -dynamiclib -Wl,-dylib -o $@ $<
%.so: %.o
	$(CC) -o $@ $< -shared
$(TMPDIR)/lib%.o: %.c
	$(CC) -c -o $@ $<

