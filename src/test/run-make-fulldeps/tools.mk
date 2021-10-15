# These deliberately use `=` and not `:=` so that client makefiles can
# augment HOST_RPATH_DIR / TARGET_RPATH_DIR.
HOST_RPATH_ENV = \
    $(LD_LIB_PATH_ENVVAR)="$(TMPDIR):$(HOST_RPATH_DIR):$($(LD_LIB_PATH_ENVVAR))"
TARGET_RPATH_ENV = \
    $(LD_LIB_PATH_ENVVAR)="$(TMPDIR):$(TARGET_RPATH_DIR):$($(LD_LIB_PATH_ENVVAR))"

RUSTC_ORIGINAL := $(RUSTC)
BARE_RUSTC := $(HOST_RPATH_ENV) '$(RUSTC)'
BARE_RUSTDOC := $(HOST_RPATH_ENV) '$(RUSTDOC)'
RUSTC := $(BARE_RUSTC) --out-dir $(TMPDIR) -L $(TMPDIR) $(RUSTFLAGS)
RUSTDOC := $(BARE_RUSTDOC) -L $(TARGET_RPATH_DIR)
ifdef RUSTC_LINKER
RUSTC := $(RUSTC) -Clinker='$(RUSTC_LINKER)'
RUSTDOC := $(RUSTDOC) -Clinker='$(RUSTC_LINKER)'
endif
#CC := $(CC) -L $(TMPDIR)
HTMLDOCCK := '$(PYTHON)' '$(S)/src/etc/htmldocck.py'
CGREP := "$(S)/src/etc/cat-and-grep.sh"

# diff with common flags for multi-platform diffs against text output
DIFF := diff -u --strip-trailing-cr

# Some of the Rust CI platforms use `/bin/dash` to run `shell` script in
# Makefiles. Other platforms, including many developer platforms, default to
# `/bin/bash`. (In many cases, `make` is actually using `/bin/sh`, but `sh`
# is configured to execute one or the other shell binary). `dash` features
# support only a small subset of `bash` features, so `dash` can be thought of as
# the lowest common denominator, and tests should be validated against `dash`
# whenever possible. Most developer platforms include `/bin/dash`, but to ensure
# tests still work when `/bin/dash`, if not available, this `SHELL` override is
# conditional:
ifndef IS_WINDOWS # dash interprets backslashes in executable paths incorrectly
ifneq (,$(wildcard /bin/dash))
SHELL := /bin/dash
endif
endif

# This is the name of the binary we will generate and run; use this
# e.g. for `$(CC) -o $(RUN_BINFILE)`.
RUN_BINFILE = $(TMPDIR)/$(1)

# RUN and FAIL are basic way we will invoke the generated binary.  On
# non-windows platforms, they set the LD_LIBRARY_PATH environment
# variable before running the binary.

RLIB_GLOB = lib$(1)*.rlib
BIN = $(1)

UNAME = $(shell uname)

ifeq ($(UNAME),Darwin)
RUN = $(TARGET_RPATH_ENV) $(RUN_BINFILE)
FAIL = $(TARGET_RPATH_ENV) $(RUN_BINFILE) && exit 1 || exit 0
DYLIB_GLOB = lib$(1)*.dylib
DYLIB = $(TMPDIR)/lib$(1).dylib
STATICLIB = $(TMPDIR)/lib$(1).a
STATICLIB_GLOB = lib$(1)*.a
else
ifdef IS_WINDOWS
RUN = PATH="$(PATH):$(TARGET_RPATH_DIR)" $(RUN_BINFILE)
FAIL = PATH="$(PATH):$(TARGET_RPATH_DIR)" $(RUN_BINFILE) && exit 1 || exit 0
DYLIB_GLOB = $(1)*.dll
DYLIB = $(TMPDIR)/$(1).dll
ifdef IS_MSVC
STATICLIB = $(TMPDIR)/$(1).lib
STATICLIB_GLOB = $(1)*.lib
else
IMPLIB = $(TMPDIR)/lib$(1).dll.a
STATICLIB = $(TMPDIR)/lib$(1).a
STATICLIB_GLOB = lib$(1)*.a
endif
BIN = $(1).exe
LLVM_FILECHECK := $(shell cygpath -u "$(LLVM_FILECHECK)")
else
RUN = $(TARGET_RPATH_ENV) $(RUN_BINFILE)
FAIL = $(TARGET_RPATH_ENV) $(RUN_BINFILE) && exit 1 || exit 0
DYLIB_GLOB = lib$(1)*.so
DYLIB = $(TMPDIR)/lib$(1).so
STATICLIB = $(TMPDIR)/lib$(1).a
STATICLIB_GLOB = lib$(1)*.a
endif
endif

ifdef IS_MSVC
COMPILE_OBJ = $(CC) -c -Fo:`cygpath -w $(1)` $(2)
COMPILE_OBJ_CXX = $(CXX) -EHs -c -Fo:`cygpath -w $(1)` $(2)
NATIVE_STATICLIB_FILE = $(1).lib
NATIVE_STATICLIB = $(TMPDIR)/$(call NATIVE_STATICLIB_FILE,$(1))
OUT_EXE=-Fe:`cygpath -w $(TMPDIR)/$(call BIN,$(1))` \
	-Fo:`cygpath -w $(TMPDIR)/$(1).obj`
else
COMPILE_OBJ = $(CC) -c -o $(1) $(2)
COMPILE_OBJ_CXX = $(CXX) -c -o $(1) $(2)
NATIVE_STATICLIB_FILE = lib$(1).a
NATIVE_STATICLIB = $(call STATICLIB,$(1))
OUT_EXE=-o $(TMPDIR)/$(1)
endif


# Extra flags needed to compile a working executable with the standard library
ifdef IS_WINDOWS
ifdef IS_MSVC
	EXTRACFLAGS := ws2_32.lib userenv.lib advapi32.lib bcrypt.lib
else
	EXTRACFLAGS := -lws2_32 -luserenv -lbcrypt
	EXTRACXXFLAGS := -lstdc++
	# So this is a bit hacky: we can't use the DLL version of libstdc++ because
	# it pulls in the DLL version of libgcc, which means that we end up with 2
	# instances of the DW2 unwinding implementation. This is a problem on
	# i686-pc-windows-gnu because each module (DLL/EXE) needs to register its
	# unwind information with the unwinding implementation, and libstdc++'s
	# __cxa_throw won't see the unwinding info we registered with our statically
	# linked libgcc.
	#
	# Now, simply statically linking libstdc++ would fix this problem, except
	# that it is compiled with the expectation that pthreads is dynamically
	# linked as a DLL and will fail to link with a statically linked libpthread.
	#
	# So we end up with the following hack: we link use static-nobundle to only
	# link the parts of libstdc++ that we actually use, which doesn't include
	# the dependency on the pthreads DLL.
	EXTRARSCXXFLAGS := -l static-nobundle=stdc++
endif
else
ifeq ($(UNAME),Darwin)
	EXTRACFLAGS := -lresolv
	EXTRACXXFLAGS := -lc++
	EXTRARSCXXFLAGS := -lc++
else
ifeq ($(UNAME),FreeBSD)
	EXTRACFLAGS := -lm -lpthread -lgcc_s
else
ifeq ($(UNAME),SunOS)
	EXTRACFLAGS := -lm -lpthread -lposix4 -lsocket -lresolv
else
ifeq ($(UNAME),OpenBSD)
	EXTRACFLAGS := -lm -lpthread -lc++abi
	RUSTC := $(RUSTC) -C linker="$(word 1,$(CC:ccache=))"
else
	EXTRACFLAGS := -lm -lrt -ldl -lpthread
	EXTRACXXFLAGS := -lstdc++
	EXTRARSCXXFLAGS := -lstdc++
endif
endif
endif
endif
endif

REMOVE_DYLIBS     = rm $(TMPDIR)/$(call DYLIB_GLOB,$(1))
REMOVE_RLIBS      = rm $(TMPDIR)/$(call RLIB_GLOB,$(1))

%.a: %.o
	$(AR) crus $@ $<
ifdef IS_MSVC
%.lib: lib%.o
	$(MSVC_LIB) -out:`cygpath -w $@` $<
else
%.lib: lib%.o
	$(AR) crus $@ $<
endif
%.dylib: %.o
	$(CC) -dynamiclib -Wl,-dylib -o $@ $<
%.so: %.o
	$(CC) -o $@ $< -shared

ifdef IS_MSVC
%.dll: lib%.o
	$(CC) $< -link -dll -out:`cygpath -w $@`
else
%.dll: lib%.o
	$(CC) -o $@ $< -shared -Wl,--out-implib=$@.a
endif

$(TMPDIR)/lib%.o: %.c
	$(call COMPILE_OBJ,$@,$<)
