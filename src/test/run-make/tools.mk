# These deliberately use `=` and not `:=` so that client makefiles can
# augment HOST_RPATH_DIR / TARGET_RPATH_DIR.
HOST_RPATH_ENV = \
    $(LD_LIB_PATH_ENVVAR)="$(TMPDIR):$(HOST_RPATH_DIR):$($(LD_LIB_PATH_ENVVAR))"
TARGET_RPATH_ENV = \
    $(LD_LIB_PATH_ENVVAR)="$(TMPDIR):$(TARGET_RPATH_DIR):$($(LD_LIB_PATH_ENVVAR))"

BARE_RUSTC := $(HOST_RPATH_ENV) $(RUSTC)
RUSTC := $(BARE_RUSTC) --out-dir $(TMPDIR) -L $(TMPDIR) $(RUSTFLAGS)
#CC := $(CC) -L $(TMPDIR)
HTMLDOCCK := $(PYTHON) $(S)/src/etc/htmldocck.py

# This is the name of the binary we will generate and run; use this
# e.g. for `$(CC) -o $(RUN_BINFILE)`.
RUN_BINFILE = $(TMPDIR)/$(1)

# RUN and FAIL are basic way we will invoke the generated binary.  On
# non-windows platforms, they set the LD_LIBRARY_PATH environment
# variable before running the binary.

RLIB_GLOB = lib$(1)*.rlib
BIN = $(1)

UNAME = $(shell uname)
ifneq (,$(findstring MINGW,$(UNAME)))
IS_WINDOWS=1
endif

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
STATICLIB = $(TMPDIR)/$(1).lib
STATICLIB_GLOB = $(1)*.lib
BIN = $(1).exe
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
NATIVE_STATICLIB_FILE = $(1).lib
NATIVE_STATICLIB = $(TMPDIR)/$(call NATIVE_STATICLIB_FILE,$(1))
OUT_EXE=-Fe:`cygpath -w $(TMPDIR)/$(call BIN,$(1))` \
	-Fo:`cygpath -w $(TMPDIR)/$(1).obj`
else
COMPILE_OBJ = $(CC) -c -o $(1) $(2)
NATIVE_STATICLIB_FILE = lib$(1).a
NATIVE_STATICLIB = $(call STATICLIB,$(1))
OUT_EXE=-o $(TMPDIR)/$(1)
endif


# Extra flags needed to compile a working executable with the standard library
ifdef IS_WINDOWS
ifdef IS_MSVC
	EXTRACFLAGS := ws2_32.lib userenv.lib shell32.lib advapi32.lib
else
	EXTRACFLAGS := -lws2_32 -luserenv
endif
else
ifeq ($(UNAME),Darwin)
else
ifeq ($(UNAME),FreeBSD)
	EXTRACFLAGS := -lm -lpthread -lgcc_s
else
ifeq ($(UNAME),Bitrig)
	EXTRACFLAGS := -lm -lpthread
	EXTRACXXFLAGS := -lc++ -lc++abi
else
ifeq ($(UNAME),OpenBSD)
	EXTRACFLAGS := -lm -lpthread
	# extend search lib for found estdc++ if build using gcc from
	# ports under OpenBSD. This is needed for:
	#  - run-make/execution-engine
	#  - run-make/issue-19371
	RUSTC := $(RUSTC) -L/usr/local/lib
else
	EXTRACFLAGS := -lm -lrt -ldl -lpthread
	EXTRACXXFLAGS := -lstdc++
endif
endif
endif
endif
endif

REMOVE_DYLIBS     = rm $(TMPDIR)/$(call DYLIB_GLOB,$(1))
REMOVE_RLIBS      = rm $(TMPDIR)/$(call RLIB_GLOB,$(1))

%.a: %.o
	ar crus $@ $<
%.lib: lib%.o
	ar crus $@ $<
%.dylib: %.o
	$(CC) -dynamiclib -Wl,-dylib -o $@ $<
%.so: %.o
	$(CC) -o $@ $< -shared

ifdef IS_MSVC
%.dll: lib%.o
	$(CC) $< -link -dll -out:`cygpath -w $@`
else
%.dll: lib%.o
	$(CC) -o $@ $< -shared
endif

$(TMPDIR)/lib%.o: %.c
	$(call COMPILE_OBJ,$@,$<)
