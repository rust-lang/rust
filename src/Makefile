######################################################################
# Auto-configuration
######################################################################

ifdef VERBOSE
  CFG_QUIET :=
  CFG_ECHO =
else
  CFG_QUIET := @
  CFG_ECHO = echo $(1)
endif

CFG_OSTYPE := $(shell uname -s)
CFG_CPUTYPE := $(shell uname -m)

ifneq ($(MAKE_RESTARTS),)
CFG_INFO := $(info cfg: make restarts: $(MAKE_RESTARTS))
endif

CFG_INFO := $(info cfg: building on $(CFG_OSTYPE) $(CFG_CPUTYPE))

CFG_GCC_CFLAGS :=
CFG_GCC_LINK_FLAGS :=
CFG_VALGRIND :=

# Disable the ocaml backend on rustboot. It's too frustrating at this
# point.
NO_OCAML_LLVM := 1

CFG_LLVM_CONFIG ?= llvm-config
CFG_BOOT_FLAGS := $(FLAGS)
CFG_RUSTC_FLAGS := -nowarn

# On Darwin, we need to run dsymutil so the debugging information ends
# up in the right place.  On other platforms, it automatically gets
# embedded into the executable, so use a no-op command.
DSYMUTIL := true

ifeq ($(CFG_OSTYPE), Linux)
  CFG_RUNTIME := librustrt.so
  CFG_STDLIB := libstd.so
  CFG_GCC_CFLAGS += -fPIC -march=i686
  CFG_GCC_LINK_FLAGS += -shared -fPIC -ldl -lpthread -lrt
  ifeq ($(CFG_CPUTYPE), x86_64)
    CFG_GCC_CFLAGS += -m32
    CFG_GCC_LINK_FLAGS += -m32
  endif
  CFG_NATIVE := 1
  CFG_UNIXY := 1
  CFG_VALGRIND := $(shell which valgrind)
  ifdef CFG_VALGRIND
    CFG_VALGRIND += --leak-check=full \
                    --quiet --vex-iropt-level=0 \
                    --suppressions=etc/x86.supp
  endif
endif

ifeq ($(CFG_OSTYPE), Darwin)
  CFG_RUNTIME := librustrt.dylib
  CFG_STDLIB := libstd.dylib
  CFG_UNIXY := 1
  CFG_GCC_LINK_FLAGS += -dynamiclib -lpthread
  # Darwin has a very blurry notion of "64 bit", and claims it's running
  # "on an i386" when the whole userspace is 64-bit and the compiler
  # emits 64-bit binaries by default. So we just force -m32 here. Smarter
  # approaches welcome!
  CFG_NATIVE := 1
  CFG_GCC_CFLAGS += -m32
  CFG_GCC_LINK_FLAGS += -m32
  DSYMUTIL := dsymutil
endif

ifneq ($(findstring MINGW,$(CFG_OSTYPE)),)
  CFG_WINDOWSY := 1
endif

ifdef CFG_WINDOWSY
  CFG_INFO := $(info cfg: windows-y environment)
  CFG_PATH_MUNGE := | sed -e 's/\\\(.\)/\/\1/g'
  CFG_FLEXLINK := $(shell which flexlink)
  ifdef CFG_FLEXLINK
    CFG_NATIVE := 1
  endif
  CFG_RUNTIME := rustrt.dll
  CFG_STDLIB := std.dll
  CFG_OBJ_SUFFIX := .o
  CFG_EXE_SUFFIX := .exe
  CFG_BOOT := ./rustboot.exe
  CFG_RUSTC := ./rustc.exe
  CFG_GCC_CFLAGS += -march=i686
  CFG_GCC_LINK_FLAGS += -shared -fPIC
  CFG_RUN_TARG = $(1)
  # FIXME: support msvc at some point
  CFG_GCC := 1
endif

ifdef CFG_UNIXY
  CFG_INFO := $(info cfg: unix-y environment)
  CFG_BOOT := ./rustboot
  CFG_RUSTC := ./rustc
  CFG_OBJ_SUFFIX := .o
  CFG_RUN_TARG = LD_LIBRARY_PATH=. $(CFG_VALGRIND) $(1)
  CFG_GCC := 1
  ifdef MINGW_CROSS
    CFG_INFO := $(info cfg: mingw-cross)
    CFG_GCC_CROSS := i586-mingw32msvc-
    CFG_BOOT_FLAGS += -t win32-x86-pe
    CFG_RUNTIME := rustrt.dll
    CFG_STDLIB := std.dll
    CFG_RUSTC := ./rustc.exe
    ifdef CFG_VALGRIND
      CFG_VALGRIND += wine
    endif
    CFG_OBJ_SUFFIX := .o
    CFG_EXE_SUFFIX := .exe
    CFG_GCC_CFLAGS := -march=i686
    CFG_GCC_LINK_FLAGS := -shared
    ifeq ($(CFG_CPUTYPE), x86_64)
      CFG_GCC_CFLAGS += -m32
      CFG_GCC_LINK_FLAGS += -m32
    endif
  endif
endif

ifdef CFG_GCC
  CFG_INFO := $(info cfg: using gcc)
  CFG_GCC_CFLAGS += -Wall -Werror -fno-rtti -fno-exceptions -g
  CFG_GCC_LINK_FLAGS += -g
  CFG_COMPILE_C = $(CFG_GCC_CROSS)g++ $(CFG_GCC_CFLAGS) -c -o $(1) $(2)
  CFG_LINK_C = $(CFG_GCC_CROSS)g++ $(CFG_GCC_LINK_FLAGS) -o $(1)
  CFG_DEPEND_C = $(CFG_GCC_CROSS)g++ $(CFG_GCC_CFLAGS) -MT "$(1)" -MM $(2)
else
  CFG_ERR := $(error please try on a system with gcc)
endif

CFG_OCAMLC_OPT := $(shell which ocamlc.opt)
ifdef CFG_OCAMLC_OPT
  $(info cfg: using ocaml native compiler)
  OPT=.opt
else
  $(info cfg: using ocaml bytecode compiler)
endif

ifdef PROFILE
  $(info cfg: building with profiling info (forcing native output))
  CFG_NATIVE := 1
  CFG_OCAMLOPT_PROFILE_FLAGS := -p
  NO_LLVM := 1
endif

ifdef DEBUG
  $(info cfg: forcing bytecode output)
  CFG_NATIVE :=
endif

ifdef CFG_NATIVE
  $(info cfg: building native compiler)
else
  $(info cfg: building bytecode compiler)
endif

ifdef NO_VALGRIND
    CFG_VALGRIND :=
endif

ifdef NO_LLVM
    CFG_LLVM_CONFIG :=
endif

ifneq ($(CFG_LLVM_CONFIG),)
  CFG_LLVM_CONFIG := $(shell which $(CFG_LLVM_CONFIG))
endif
ifneq ($(CFG_LLVM_CONFIG),)
  CFG_LLVM_VERSION := $(shell $(CFG_LLVM_CONFIG) --version)
  $(info cfg: found llvm-config at $(CFG_LLVM_CONFIG))
  CFG_LLVM_ALLOWED_VERSIONS := 2.8svn 2.8 2.9svn
  ifneq ($(findstring $(CFG_LLVM_VERSION),$(CFG_LLVM_ALLOWED_VERSIONS)),)
    $(info cfg: using LLVM version $(CFG_LLVM_VERSION))
  else
    CFG_LLVM_CONFIG :=
    $(info cfg: incompatible LLVM version $(CFG_LLVM_VERSION), \
      expected one of $(CFG_LLVM_ALLOWED_VERSIONS))
  endif
endif
ifneq ($(CFG_LLVM_CONFIG),)
  CFG_OCAML_LIBPATH := $(lastword \
    $(shell ocamlc$(OPT) -config | grep standard_library:))
  CFG_OCAML_LLVM := $(shell \
      for path in $(shell $(CFG_LLVM_CONFIG) --libdir)/ocaml \
          $(CFG_OCAML_LIBPATH) \
          $(CFG_OCAML_LIBPATH)/llvm \
          $(CFG_OCAML_LIBPATH)/llvm-$(CFG_LLVM_VERSION) ; do \
        if test -e $${path}/llvm.cma; then echo $${path}; break; fi \
      done)
  ifneq ($(CFG_OCAML_LLVM),)
    $(info cfg: found LLVM ocaml bindings in $(CFG_OCAML_LLVM))
  else
    CFG_LLVM_CONFIG := $(info cfg: LLVM ocaml bindings not found)
  endif
endif

ifdef NO_OCAML_LLVM
    $(info cfg: disabling LLVM ocaml bindings)
    CFG_OCAML_LLVM :=
endif

ifdef CFG_LLVM_CONFIG
  LLC := "$(shell $(CFG_LLVM_CONFIG) --bindir)/llc"
  CFG_LLC_CFLAGS := -march=x86
  LLVM-DIS := "$(shell $(CFG_LLVM_CONFIG) --bindir)/llvm-dis"
  ifneq ($(CFG_OCAML_LLVM),)
    VARIANT=llvm
    LLVM_LIBS := llvm.cma llvm_bitwriter.cma
    LLVM_NATIVE_LIBS := llvm.cmxa llvm_bitwriter.cmxa
    LLVM_CLIBS := $(shell for c in `$(CFG_LLVM_CONFIG) --ldflags --libs` \
      -lllvm -lllvm_bitwriter; do echo -cclib && echo $$c; done | xargs echo)
    LLVM_INCS := -I boot/llvm -I $(CFG_OCAML_LLVM)
    LLVM_MLS := $(addprefix boot/llvm/, llabi.ml llasm.ml llfinal.ml \
      lltrans.ml llemit.ml)
  else
    VARIANT=x86
    LLVM_CLIBS :=
    LLVM_INCS :=
    LLVM_MLS :=
    $(info cfg: disabling LLVM ocaml backend)
  endif
else
  VARIANT=x86
  LLVM_CLIBS :=
  LLVM_INCS :=
  LLVM_MLS :=
  $(info cfg: disabling LLVM backend)
endif

MKFILES := Makefile

# Add current directory to library search path of CFG_BOOT command line.

CFG_BOOT_FLAGS += -L .

######################################################################
# Boot targets and rules
######################################################################

ML_DEP_INCS := -I boot/fe -I boot/me -I boot/be -I boot/driver/$(VARIANT) \
  -I boot/driver -I boot/util -I boot/llvm
ML_INCS := $(ML_DEP_INCS) $(LLVM_INCS)
ML_LIBS := unix.cma nums.cma bigarray.cma
ML_NATIVE_LIBS := unix.cmxa nums.cmxa bigarray.cmxa
OCAMLC_FLAGS := -g $(ML_INCS) -w Ael -warn-error Ael
OCAMLOPT_FLAGS := $(ML_INCS) -w Ael -warn-error Ael \
                    $(CFG_OCAMLOPT_PROFILE_FLAGS)

ifdef CFG_LLVM_CONFIG
  ML_LIBS += $(LLVM_LIBS) -custom -cclib -lstdc++ $(LLVM_CLIBS)
  ML_NATIVE_LIBS += $(LLVM_NATIVE_LIBS) -cclib -lstdc++ $(LLVM_CLIBS)
endif

# List them in link order.
# Nobody calculates the link-order DAG automatically, sadly.

UTIL_BOT_MLS := $(addprefix boot/util/, version.ml fmt.ml common.ml bits.ml)
DRIVER_BOT_MLS := $(addprefix boot/driver/, session.ml)
BE_MLS := $(addprefix boot/be/, x86.ml ra.ml pe.ml elf.ml \
          macho.ml)
IL_MLS := $(addprefix boot/be/, asm.ml il.ml abi.ml)
ME_MLS := $(addprefix boot/me/, walk.ml semant.ml resolve.ml alias.ml \
            simplify.ml type.ml dead.ml layer.ml effect.ml typestate.ml \
            loop.ml layout.ml transutil.ml trans.ml dwarf.ml)
FE_MLS := $(addprefix boot/fe/, ast.ml token.ml lexer.ml parser.ml \
            extfmt.ml pexp.ml item.ml cexp.ml fuzz.ml)
DRIVER_TOP_MLS := $(addprefix boot/driver/, lib.ml $(VARIANT)/glue.ml main.ml)

BOOT_MLS := $(UTIL_BOT_MLS) $(DRIVER_BOT_MLS) $(FE_MLS) $(IL_MLS) $(ME_MLS) \
  $(BE_MLS) $(LLVM_MLS) $(DRIVER_TOP_MLS)
BOOT_CMOS := $(BOOT_MLS:.ml=.cmo)
BOOT_CMXS := $(BOOT_MLS:.ml=.cmx)
BOOT_OBJS := $(BOOT_MLS:.ml=.o)
BOOT_CMIS := $(BOOT_MLS:.ml=.cmi)

RUNTIME_CS := rt/sync/timer.cpp \
              rt/sync/sync.cpp \
              rt/sync/lock_and_signal.cpp \
              rt/rust.cpp \
              rt/rust_builtin.cpp \
              rt/rust_crate.cpp \
              rt/rust_crate_cache.cpp \
              rt/rust_crate_reader.cpp \
              rt/rust_comm.cpp \
              rt/rust_dom.cpp \
              rt/rust_task.cpp \
              rt/rust_task_list.cpp \
              rt/rust_proxy.cpp \
              rt/rust_chan.cpp \
              rt/rust_port.cpp \
              rt/rust_upcall.cpp \
              rt/rust_log.cpp \
              rt/rust_message.cpp \
              rt/rust_timer.cpp \
              rt/circular_buffer.cpp \
              rt/isaac/randport.cpp \
              rt/rust_srv.cpp \
              rt/rust_kernel.cpp \
              rt/memory_region.cpp \
              rt/test/rust_test_harness.cpp \
              rt/test/rust_test_runtime.cpp \
              rt/test/rust_test_util.cpp

RUNTIME_HDR := rt/globals.h \
               rt/rust.h \
               rt/rust_dwarf.h \
               rt/rust_internal.h \
               rt/rust_util.h \
               rt/rust_chan.h \
               rt/rust_port.h \
               rt/rust_dom.h \
               rt/rust_task.h \
               rt/rust_task_list.h \
               rt/rust_proxy.h \
               rt/rust_log.h \
               rt/rust_message.h \
               rt/circular_buffer.h \
               rt/util/array_list.h \
               rt/util/indexed_list.h \
               rt/util/synchronized_indexed_list.h \
               rt/util/hash_map.h \
               rt/sync/sync.h \
               rt/sync/timer.h \
               rt/sync/lock_free_queue.h \
               rt/rust_srv.h \
               rt/rust_kernel.h \
               rt/memory_region.h \
               rt/memory.h \
               rt/test/rust_test_harness.h \
               rt/test/rust_test_runtime.h \
               rt/test/rust_test_util.h

RUNTIME_INCS := -Irt/isaac -Irt/uthash
RUNTIME_OBJS := $(RUNTIME_CS:.cpp=$(CFG_OBJ_SUFFIX))
RUNTIME_LIBS := $(CFG_RUNTIME_LIBS)

STDLIB_CRATE := lib/std.rc
STDLIB_INPUTS := $(wildcard lib/*.rc lib/*.rs lib/*/*.rs)
COMPILER_CRATE := comp/rustc.rc
COMPILER_INPUTS := $(wildcard comp/*.rc comp/*.rs comp/*/*.rs)

GENERATED := boot/fe/lexer.ml boot/util/version.ml

all: $(CFG_RUSTC) $(MKFILES) $(GENERATED)

boot/util/version.ml: Makefile
	$(CFG_QUIET)git log -1 \
      --format='let version = "prerelease (%h %ci)";;' >$@ || exit 1

loc:
	$(CFG_QUIET)wc -l $(BOOT_MLS) $(RUNTIME_CS) $(RUNTIME_HDR)

$(CFG_RUNTIME): $(RUNTIME_OBJS) $(MKFILES) $(RUNTIME_HDR)
	@$(call CFG_ECHO, compile: $<)
	$(CFG_QUIET)$(call CFG_LINK_C, $@) $(RUNTIME_OBJS)

$(CFG_STDLIB): $(STDLIB_CRATE) $(CFG_BOOT) $(MKFILES)
	@$(call CFG_ECHO, compile: $<)
	$(CFG_QUIET)OCAMLRUNPARAM="b1" $(CFG_BOOT) $(CFG_BOOT_FLAGS) \
      -shared -o $@ $(STDLIB_CRATE)

%$(CFG_OBJ_SUFFIX): %.cpp $(MKFILES)
	@$(call CFG_ECHO, compile: $<)
	$(CFG_QUIET)$(call CFG_COMPILE_C, $@, $(RUNTIME_INCS)) $<

ifdef CFG_NATIVE
$(CFG_BOOT): $(BOOT_CMXS) $(MKFILES)
	@$(call CFG_ECHO, compile: $<)
	$(CFG_QUIET)ocamlopt$(OPT) -o $@ $(OCAMLOPT_FLAGS) $(ML_NATIVE_LIBS) \
      $(BOOT_CMXS)
else
$(CFG_BOOT): $(BOOT_CMOS) $(MKFILES)
	@$(call CFG_ECHO, compile: $<)
	$(CFG_QUIET)ocamlc$(OPT) -o $@ $(OCAMLC_FLAGS) $(ML_LIBS) $(BOOT_CMOS)
endif

%.cmo: %.ml $(MKFILES)
	@$(call CFG_ECHO, compile: $<)
	$(CFG_QUIET)ocamlc$(OPT) -c -o $@ $(OCAMLC_FLAGS) $<

%.cmo: %.cmi $(MKFILES)

%.cmx %.o: %.ml $(MKFILES)
	@$(call CFG_ECHO, compile: $<)
	$(CFG_QUIET)ocamlopt$(OPT) -c -o $@ $(OCAMLOPT_FLAGS) $<

%.ml: %.mll $(MKFILES)
	@$(call CFG_ECHO, lex-gen: $<)
	$(CFG_QUIET)ocamllex$(OPT) -q -o $@ $<


######################################################################
# Main compiler targets and rules
######################################################################

$(CFG_RUSTC): $(COMPILER_INPUTS) $(CFG_BOOT) $(CFG_RUNTIME) $(CFG_STDLIB)
	@$(call CFG_ECHO, compile: $<)
	$(CFG_QUIET)OCAMLRUNPARAM="b1" $(CFG_BOOT) $(CFG_BOOT_FLAGS) \
      -minimal -o $@ $<
	$(CFG_QUIET)chmod 0755 $@

self: $(CFG_RUSTC)
	@$(call CFG_ECHO, compile: $<)
	$(CFG_QUIET)$(call CFG_RUN_TARG, $(CFG_RUSTC)) $(COMPILER_CRATE)


######################################################################
# Testing
######################################################################

# Temporarily xfail tests broken by the nominal-tags change.

NOMINAL_TAG_XFAILS := test/run-pass/mlist.rs

# Temporarily xfail tests broken by the constant-tags change.

CONST_TAG_XFAILS := test/run-pass/generic-tag.rs

# Temporarily xfail some of the task tests, while debugging the
# overhauled inter-domain messaging system.

TASK_XFAILS := test/run-pass/task-comm-8.rs \
               test/run-pass/task-comm-10.rs \
               test/run-pass/task-comm-15.rs \
               test/run-pass/task-comm-12.rs \
               test/run-pass/task-comm-2.rs \
               test/run-pass/task-comm-9.rs \
               test/run-pass/task-life-0.rs \
               test/run-pass/alt-type-simple.rs \
               test/run-pass/many.rs

TEST_XFAILS_X86 :=  $(TASK_XFAILS) \
                    $(NOMINAL_TAG_XFAILS) \
                    $(CONST_TAG_XFAILS) \
                    test/run-pass/child-outlives-parent.rs \
                    test/run-pass/clone-with-exterior.rs \
                    test/run-pass/constrained-type.rs \
                    test/run-pass/destructor-ordering.rs \
                    test/run-pass/obj-as.rs \
                    test/run-pass/vec-slice.rs \
                    test/run-pass/fn-lval.rs \
                    test/run-pass/generic-fn-infer.rs \
                    test/run-pass/generic-recursive-tag.rs \
                    test/run-pass/iter-ret.rs \
                    test/run-pass/lib-io.rs \
                    test/run-pass/mlist-cycle.rs \
                    test/run-pass/obj-as.rs \
                    test/run-pass/task-comm.rs \
                    test/run-pass/task-comm-3.rs \
                    test/run-pass/vec-slice.rs \
                    test/run-pass/while-and-do-while.rs \
                    test/run-fail/task-comm-14.rs \
                    test/compile-fail/bad-recv.rs \
                    test/compile-fail/bad-send.rs \
                    test/compile-fail/infinite-vec-type-recursion.rs \
                    test/compile-fail/writing-through-read-alias.rs

# Most tests fail on the boot-llvm backend, so we list the run-pass
# xfails inside-out, by listing those that *don't* fail. A bit
# double-negative-y but it's a bit unwieldy the other way just now.

TEST_XFAILS_LLVM := $(TASK_XFAILS) \
                    $(NOMINAL_TAG_XFAILS) \
                    $(CONST_TAG_XFAILS) \
                    $(filter-out \
                      $(addprefix test/run-pass/, \
                        arith-0.rs \
                        arith-2.rs \
                        box.rs \
                        char.rs \
                        dead-code-one-arm-if.rs \
                        div-mod.rs \
                        drop-on-ret.rs \
                        else-if.rs \
                        fact.rs \
                        generic-fn-twice.rs \
                        generic-type.rs \
                        generic-type-synonym.rs \
                        hello.rs \
                        inner-module.rs \
                        int.rs \
                        item-name-overload.rs \
                        multiline-comment.rs \
                        native.rs \
                        native-mod.rs \
                        native-opaque-type.rs \
                        opeq.rs \
                        return-nil.rs \
                        syntax-extension-shell.rs \
                        tup.rs), \
                      $(wildcard test/run-pass/*.rs) ) \
                    $(addprefix test/run-fail/, \
                      explicit-fail.rs \
                      fail.rs \
                      linked-failure.rs \
                      non-exhaustive-match.rs \
                      pred.rs \
                      str-overrun.rs \
                      vec-overrun.rs \
                      vec-underrun.rs \
                      task-comm-14.rs \
                     ) \
                    $(addprefix test/compile-fail/, \
                      bad-recv.rs \
                      bad-send.rs \
                      infinite-vec-type-recursion.rs \
                      rec-missing-fields.rs \
                      writing-through-read-alias.rs \
                     )

# Same strategy here for the time being: just list the ones that
# work and assume the others don't. Invert this when we're closer
# to actually bootstrapping.

TEST_XFAILS_SELF := $(filter-out \
                      $(addprefix test/run-pass/, \
                        alt-pattern-simple.rs \
                        alt-tag.rs \
                        arith-0.rs \
                        arith-1.rs \
                        arith-2.rs \
                        bitwise.rs \
                        bool-not.rs \
                        box.rs \
                        box-in-tup.rs \
                        cast.rs \
                        char.rs \
                        complex.rs \
                        dead-code-one-arm-if.rs \
                        deep.rs \
                        deref.rs \
                        div-mod.rs \
                        drop-on-ret.rs \
                        fact.rs \
                        hello.rs \
                        int.rs \
                        i32-sub.rs \
                        i8-incr.rs \
                        item-name-overload.rs \
                        large-records.rs \
                        lazy-init.rs \
                        lazy-and-or.rs \
                        multiline-comment.rs \
                        operator-associativity.rs \
                        output-slot-variants.rs \
                        readalias.rs \
                        rec.rs \
                        rec-auto.rs \
                        rec-tup.rs \
                        return-nil.rs \
                        simple-obj.rs \
                        type-in-nested-module.rs \
                        tup.rs \
                        u32-decr.rs \
                        u8-incr.rs \
                        u8-incr-decr.rs \
                        uint.rs \
                        unit.rs \
                        tag.rs \
                        vec.rs \
                        vec-drop.rs \
                        vec-in-tup.rs \
                        vec-late-init.rs \
                        while-and-do-while.rs \
                        writealias.rs \
                        ) \
                      $(addprefix test/compile-fail/, \
                        arg-count-mismatch.rs \
                        arg-type-mismatch.rs \
                        while-type-error.rs \
                        ), \
                      $(wildcard test/*/*.rs test/*/*.rc))


ifdef MINGW_CROSS
TEST_XFAILS_X86 += test/run-pass/native-mod.rc
TEST_XFAILS_LLVM += test/run-pass/native-mod.rc
TEST_XFAILS_SELF += test/run-pass/native-mod.rc
endif
ifdef CFG_WINDOWSY
TEST_XFAILS_X86 += test/run-pass/native-mod.rc
TEST_XFAILS_LLVM += test/run-pass/native-mod.rc
TEST_XFAILS_SELF += test/run-pass/native-mod.rc
endif

RPASS_RC := $(wildcard test/run-pass/*.rc)
RPASS_RS := $(wildcard test/run-pass/*.rs)
RFAIL_RC := $(wildcard test/run-fail/*.rc)
RFAIL_RS := $(wildcard test/run-fail/*.rs)
CFAIL_RC := $(wildcard test/compile-fail/*.rc)
CFAIL_RS := $(wildcard test/compile-fail/*.rs)

ifdef CHECK_XFAILS
TEST_RPASS_CRATES_X86 := $(filter $(TEST_XFAILS_X86), $(RPASS_RC))
TEST_RPASS_CRATES_LLVM := $(filter $(TEST_XFAILS_LLVM), $(RPASS_RC))
TEST_RPASS_CRATES_SELF := $(filter $(TEST_XFAILS_SELF), $(RPASS_RC))
TEST_RPASS_SOURCES_X86 := $(filter $(TEST_XFAILS_X86), $(RPASS_RS))
TEST_RPASS_SOURCES_LLVM := $(filter $(TEST_XFAILS_LLVM), $(RPASS_RS))
TEST_RPASS_SOURCES_SELF := $(filter $(TEST_XFAILS_SELF), $(RPASS_RS))
else
TEST_RPASS_CRATES_X86 := $(filter-out $(TEST_XFAILS_X86), $(RPASS_RC))
TEST_RPASS_CRATES_LLVM := $(filter-out $(TEST_XFAILS_LLVM), $(RPASS_RC))
TEST_RPASS_CRATES_SELF := $(filter-out $(TEST_XFAILS_SELF), $(RPASS_RC))
TEST_RPASS_SOURCES_X86 := $(filter-out $(TEST_XFAILS_X86), $(RPASS_RS))
TEST_RPASS_SOURCES_LLVM := $(filter-out $(TEST_XFAILS_LLVM), $(RPASS_RS))
TEST_RPASS_SOURCES_SELF := $(filter-out $(TEST_XFAILS_SELF), $(RPASS_RS))
endif

TEST_RPASS_EXES_X86 := \
              $(TEST_RPASS_CRATES_X86:.rc=-boot.x86$(CFG_EXE_SUFFIX)) \
              $(TEST_RPASS_SOURCES_X86:.rs=-boot.x86$(CFG_EXE_SUFFIX))
TEST_RPASS_EXES_LLVM := \
              $(TEST_RPASS_CRATES_LLVM:.rc=-boot.llvm$(CFG_EXE_SUFFIX)) \
              $(TEST_RPASS_SOURCES_LLVM:.rs=-boot.llvm$(CFG_EXE_SUFFIX))
TEST_RPASS_EXES_SELF := \
              $(TEST_RPASS_CRATES_SELF:.rc=.llvm$(CFG_EXE_SUFFIX)) \
              $(TEST_RPASS_SOURCES_SELF:.rs=.llvm$(CFG_EXE_SUFFIX))

TEST_RPASS_OUTS_X86  := \
                    $(TEST_RPASS_EXES_X86:.x86$(CFG_EXE_SUFFIX)=.x86.out)
TEST_RPASS_OUTS_LLVM := \
                    $(TEST_RPASS_EXES_LLVM:.llvm$(CFG_EXE_SUFFIX)=.llvm.out)
TEST_RPASS_OUTS_SELF := \
                    $(TEST_RPASS_EXES_SELF:.llvm$(CFG_EXE_SUFFIX)=.llvm.out)

TEST_RPASS_TMPS_X86  := \
    $(TEST_RPASS_EXES_X86:.x86$(CFG_EXE_SUFFIX)=.x86$(CFG_EXE_SUFFIX).tmp)
TEST_RPASS_TMPS_LLVM := \
    $(TEST_RPASS_EXES_LLVM:.llvm$(CFG_EXE_SUFFIX)=.llvm$(CFG_EXE_SUFFIX).tmp)
TEST_RPASS_TMPS_SELF := \
    $(TEST_RPASS_EXES_SELF:.llvm$(CFG_EXE_SUFFIX)=.llvm$(CFG_EXE_SUFFIX).tmp)


TEST_RFAIL_CRATES_X86 := $(filter-out $(TEST_XFAILS_X86), $(RFAIL_RC))
TEST_RFAIL_CRATES_LLVM := $(filter-out $(TEST_XFAILS_LLVM), $(RFAIL_RC))
TEST_RFAIL_CRATES_SELF := $(filter-out $(TEST_XFAILS_SELF), $(RFAIL_RC))
TEST_RFAIL_SOURCES_X86 := $(filter-out $(TEST_XFAILS_X86), $(RFAIL_RS))
TEST_RFAIL_SOURCES_LLVM := $(filter-out $(TEST_XFAILS_LLVM), $(RFAIL_RS))
TEST_RFAIL_SOURCES_SELF := $(filter-out $(TEST_XFAILS_SELF), $(RFAIL_RS))

TEST_RFAIL_EXES_X86 := \
              $(TEST_RFAIL_CRATES_X86:.rc=-boot.x86$(CFG_EXE_SUFFIX)) \
              $(TEST_RFAIL_SOURCES_X86:.rs=-boot.x86$(CFG_EXE_SUFFIX))
TEST_RFAIL_EXES_LLVM := \
              $(TEST_RFAIL_CRATES_LLVM:.rc=-boot.llvm$(CFG_EXE_SUFFIX)) \
              $(TEST_RFAIL_SOURCES_LLVM:.rs=-boot.llvm$(CFG_EXE_SUFFIX))
TEST_RFAIL_EXES_SELF := \
              $(TEST_RFAIL_CRATES_SELF:.rc=-boot.llvm$(CFG_EXE_SUFFIX)) \
              $(TEST_RFAIL_SOURCES_SELF:.rs=-boot.llvm$(CFG_EXE_SUFFIX))

TEST_RFAIL_OUTS_X86  := \
                    $(TEST_RFAIL_EXES_X86:.x86$(CFG_EXE_SUFFIX)=.x86.out)
TEST_RFAIL_OUTS_LLVM := \
                    $(TEST_RFAIL_EXES_LLVM:.llvm$(CFG_EXE_SUFFIX)=.llvm.out)
TEST_RFAIL_OUTS_SELF := \
                    $(TEST_RFAIL_EXES_SELF:.llvm$(CFG_EXE_SUFFIX)=.llvm.out)

TEST_RFAIL_TMPS_X86  := \
    $(TEST_RFAIL_EXES_X86:.x86$(CFG_EXE_SUFFIX)=.x86$(CFG_EXE_SUFFIX).tmp)
TEST_RFAIL_TMPS_LLVM := \
    $(TEST_RFAIL_EXES_LLVM:.llvm$(CFG_EXE_SUFFIX)=.llvm$(CFG_EXE_SUFFIX).tmp)
TEST_RFAIL_TMPS_SELF := \
    $(TEST_RFAIL_EXES_SELF:.llvm$(CFG_EXE_SUFFIX)=.llvm$(CFG_EXE_SUFFIX).tmp)


TEST_CFAIL_CRATES_X86 := $(filter-out $(TEST_XFAILS_X86), $(CFAIL_RC))
TEST_CFAIL_CRATES_LLVM := $(filter-out $(TEST_XFAILS_LLVM), $(CFAIL_RC))
TEST_CFAIL_CRATES_SELF := $(filter-out $(TEST_XFAILS_SELF), $(CFAIL_RC))
TEST_CFAIL_SOURCES_X86 := $(filter-out $(TEST_XFAILS_X86), $(CFAIL_RS))
TEST_CFAIL_SOURCES_LLVM := $(filter-out $(TEST_XFAILS_LLVM), $(CFAIL_RS))
TEST_CFAIL_SOURCES_SELF := $(filter-out $(TEST_XFAILS_SELF), $(CFAIL_RS))

TEST_CFAIL_EXES_X86 := \
              $(TEST_CFAIL_CRATES_X86:.rc=-boot.x86$(CFG_EXE_SUFFIX)) \
              $(TEST_CFAIL_SOURCES_X86:.rs=-boot.x86$(CFG_EXE_SUFFIX))
TEST_CFAIL_EXES_LLVM := \
              $(TEST_CFAIL_CRATES_LLVM:.rc=-boot.llvm$(CFG_EXE_SUFFIX)) \
              $(TEST_CFAIL_SOURCES_LLVM:.rs=-boot.llvm$(CFG_EXE_SUFFIX))
TEST_CFAIL_EXES_SELF := \
              $(TEST_CFAIL_CRATES_SELF:.rc=.llvm$(CFG_EXE_SUFFIX)) \
              $(TEST_CFAIL_SOURCES_SELF:.rs=.llvm$(CFG_EXE_SUFFIX))

TEST_CFAIL_OUTS_X86 := \
                $(TEST_CFAIL_EXES_X86:.x86$(CFG_EXE_SUFFIX)=.x86.out)
TEST_CFAIL_OUTS_LLVM := \
                $(TEST_CFAIL_EXES_LLVM:.llvm$(CFG_EXE_SUFFIX)=.llvm.out)
TEST_CFAIL_OUTS_SELF := \
                $(TEST_CFAIL_EXES_SELF:.llvm$(CFG_EXE_SUFFIX)=.llvm.out)

TEST_CFAIL_TMPS_X86 := \
    $(TEST_CFAIL_EXES_X86:.x86$(CFG_EXE_SUFFIX)=.x86$(CFG_EXE_SUFFIX).tmp)
TEST_CFAIL_TMPS_LLVM := \
    $(TEST_CFAIL_EXES_LLVM:.llvm$(CFG_EXE_SUFFIX)=.llvm$(CFG_EXE_SUFFIX).tmp)
TEST_CFAIL_TMPS_SELF := \
    $(TEST_CFAIL_EXES_SELF:.llvm$(CFG_EXE_SUFFIX)=.llvm$(CFG_EXE_SUFFIX).tmp)


ALL_TEST_CRATES := $(TEST_CFAIL_CRATES_X86) \
                   $(TEST_RFAIL_CRATES_X86) \
                   $(TEST_RPASS_CRATES_X86)

ALL_TEST_SOURCES := $(TEST_CFAIL_SOURCES_X86) \
                    $(TEST_RFAIL_SOURCES_X86) \
                    $(TEST_RPASS_SOURCES_X86)

ALL_TEST_INPUTS := $(wildcard test/*/*.rs test/*/*/*.rs test/*/*.rc)


check_nocompile: $(TEST_CFAIL_OUTS_X86)

check: tidy \
       $(TEST_RPASS_EXES_X86) $(TEST_RFAIL_EXES_X86) \
       $(TEST_RPASS_OUTS_X86) $(TEST_RFAIL_OUTS_X86) \
       $(TEST_CFAIL_OUTS_X86)

compile-check: tidy \
       $(TEST_RPASS_EXES_X86) $(TEST_RFAIL_EXES_X86)


ifdef CFG_LLVM_CONFIG

ifneq ($(CFG_OCAML_LLVM),)

ALL_TEST_CRATES += $(TEST_CFAIL_CRATES_LLVM) \
                   $(TEST_RFAIL_CRATES_LLVM) \
                   $(TEST_RPASS_CRATES_LLVM)

ALL_TEST_SOURCES += $(TEST_CFAIL_SOURCES_LLVM) \
                    $(TEST_RFAIL_SOURCES_LLVM) \
                    $(TEST_RPASS_SOURCES_LLVM)

check_nocompile: $(TEST_CFAIL_OUTS_LLVM)

check:	tidy \
        $(TEST_RPASS_EXES_LLVM) $(TEST_RFAIL_EXES_LLVM) \
        $(TEST_RPASS_OUTS_LLVM) $(TEST_RFAIL_OUTS_LLVM) \
        $(TEST_CFAIL_OUTS_LLVM)

compile-check: tidy \
       $(TEST_RPASS_EXES_LLVM) $(TEST_RFAIL_EXES_LLVM)

endif

ALL_TEST_CRATES += $(TEST_CFAIL_CRATES_SELF) \
                   $(TEST_RFAIL_CRATES_SELF) \
                   $(TEST_RPASS_CRATES_SELF)

ALL_TEST_SOURCES += $(TEST_CFAIL_SOURCES_SELF) \
                    $(TEST_RFAIL_SOURCES_SELF) \
                    $(TEST_RPASS_SOURCES_SELF)

check_nocompile: $(TEST_CFAIL_OUTS_SELF)

check:  tidy \
        $(TEST_RPASS_EXES_SELF) $(TEST_RFAIL_EXES_SELF) \
        $(TEST_RPASS_OUTS_SELF) $(TEST_RFAIL_OUTS_SELF) \
        $(TEST_CFAIL_OUTS_SELF)

compile-check: tidy \
       $(TEST_RPASS_EXES_LLVM) $(TEST_RFAIL_EXES_LLVM) \
       $(TEST_RPASS_EXES_SELF) $(TEST_RFAIL_EXES_SELF)

endif

BREQ := $(CFG_BOOT) $(CFG_RUNTIME) $(CFG_STDLIB)
SREQ := $(CFG_RUSTC) $(CFG_RUNTIME) $(CFG_STDLIB)
BOOT := $(CFG_QUIET)OCAMLRUNPARAM="b1" $(CFG_BOOT) $(CFG_BOOT_FLAGS)
SELF := $(CFG_QUIET)$(CFG_RUSTC) $(CFG_RUSTC_FLAGS)

# Cancel the implicit .out rule in GNU make.
%.out: %

%.out: %.out.tmp
	$(CFG_QUIET)mv $< $@

test/run-pass/%.out.tmp: test/run-pass/%$(CFG_EXE_SUFFIX) $(CFG_RUNTIME)
	$(CFG_QUIET)rm -f $<.tmp
	@$(call CFG_ECHO, run: $<)
	$(CFG_QUIET)$(call CFG_RUN_TARG, $<) > $@

test/run-fail/%-boot.x86.out.tmp: test/run-fail/%-boot.x86$(CFG_EXE_SUFFIX) \
                                  $(CFG_RUNTIME)
	$(CFG_QUIET)rm -f $<.tmp
	@$(call CFG_ECHO, run: $<)
	$(CFG_QUIET)rm -f $@
	$(CFG_QUIET)$(call CFG_RUN_TARG, $<) >$@ 2>&1 ; X=$$? ; \
      if [ $$X -eq 0 ] ; then exit 1 ; else exit 0 ; fi
	$(CFG_QUIET)grep --text --quiet \
      "`awk -F: '/error-pattern/ { print $$2 }' \
        $(subst -boot.x86.out.tmp,.rs,$@) \
        | tr -d '\n\r'`" $@

test/compile-fail/%-boot.x86.out.tmp: test/compile-fail/%.rs $(BREQ)
	@$(call CFG_ECHO, compile [boot x86]: $<)
	$(CFG_QUIET)rm -f $@
	$(BOOT) -o $(@:.out=$(CFG_EXE_SUFFIX)) $< >$@ 2>&1 || true
	$(CFG_QUIET)grep --text --quiet \
      "`awk -F: '/error-pattern/ { print $$2 }' $< | tr -d '\n\r'`" $@

test/compile-fail/%-boot.llvm.out.tmp: test/compile-fail/%.rs $(BREQ)
	@$(call CFG_ECHO, compile [boot llvm]: $<)
	$(CFG_QUIET)rm -f $@
	$(BOOT) -o $(@:.out=$(CFG_EXE_SUFFIX)) $< >$@ 2>&1 || true
	$(CFG_QUIET)grep --text --quiet \
      "`awk -F: '/error-pattern/ { print $$2 }' $< | tr -d '\n\r'`" $@

test/compile-fail/%.llvm.out.tmp: test/compile-fail/%.rs $(BREQ)
	@$(call CFG_ECHO, compile [rustc]: $<)
	$(CFG_QUIET)rm -f $@
	$(BOOT) -o $(@:.out=$(CFG_EXE_SUFFIX)) $< >$@ 2>&1 || true
	$(CFG_QUIET)grep --text --quiet \
      "`awk -F: '/error-pattern/ { print $$2 }' $< | tr -d '\n\r'`" $@

test/run-pass/%-boot.x86$(CFG_EXE_SUFFIX): test/run-pass/%.rc $(BREQ)
	@$(call CFG_ECHO, compile [boot x86]: $<)
	$(BOOT) -o $@ $<

test/bench/shootout/%-boot.x86$(CFG_EXE_SUFFIX): \
  test/bench/shootout/%.rs $(BREQ)
	@$(call CFG_ECHO, compile [boot x86]: $<)
	$(BOOT) -o $@ $<

%.ll: %.bc
	@$(call CFG_ECHO, dis [llvm]: $<)
	$(CFG_QUIET)$(LLVM-DIS) -o $@ $<


%.s: %.bc
	@$(call CFG_ECHO, compile [llvm]: $<)
	$(CFG_QUIET)$(LLC) $(CFG_LLC_CFLAGS) -o $@ $<

%.o: %.s
	@$(call CFG_ECHO, assemble [llvm]: $<)
	$(CFG_QUIET)gcc $(CFG_GCC_CFLAGS) -o $@ -c $<

%.llvm$(CFG_EXE_SUFFIX): %.o $(CFG_RUNTIME)
	@$(call CFG_ECHO, link [llvm]: $<)
	$(CFG_QUIET)gcc $(CFG_GCC_CFLAGS) -o $@ $< -L. -lrustrt
	@# dsymutil sometimes fails or prints a warning, but the
	@# program still runs.  Since it simplifies debugging other
	@# programs, I\'ll live with the noise.
	-$(CFG_QUIET)$(DSYMUTIL) $@

test/run-pass/%-boot.bc: test/run-pass/%.rc $(BREQ)
	@$(call CFG_ECHO, compile [boot llvm]: $<)
	$(BOOT) -o $@ -llvm $<

test/run-pass/%.bc: test/run-pass/%.rc $(SREQ)
	@$(call CFG_ECHO, compile [rustc]: $<)
	$(SELF) -o $@ $<

test/run-pass/%-boot.x86$(CFG_EXE_SUFFIX): test/run-pass/%.rs $(BREQ)
	@$(call CFG_ECHO, compile [boot x86]: $<)
	$(BOOT) -o $@ $<

test/run-pass/%-boot.bc: test/run-pass/%.rs $(BREQ)
	@$(call CFG_ECHO, compile [boot llvm]: $<)
	$(BOOT) -o $@ -llvm $<

test/run-pass/%.bc: test/run-pass/%.rs $(SREQ)
	@$(call CFG_ECHO, compile [rustc]: $<)
	$(SELF) -o $@ $<

test/run-fail/%-boot.x86$(CFG_EXE_SUFFIX): test/run-fail/%.rs $(BREQ)
	@$(call CFG_ECHO, compile [boot x86]: $<)
	$(BOOT) -o $@ $<

test/run-fail/%-boot.bc: test/run-fail/%.rs $(BREQ)
	@$(call CFG_ECHO, compile [boot llvm]: $<)
	$(BOOT) -o $@ -llvm $<

test/run-fail/%.bc: test/run-fail/%.rs $(BREQ)
	@$(call CFG_ECHO, compile [rustc]: $<)
	$(SELF) -o $@ $<


######################################################################
# Auto-dependency
######################################################################

ML_DEPFILES := $(BOOT_MLS:%.ml=%.d)
C_DEPFILES := $(RUNTIME_CS:%.cpp=%.d)

%.d: %.cpp $(MKFILES)
	@$(call CFG_ECHO, dep: $<)
	$(CFG_QUIET)$(call CFG_DEPEND_C, $@ \
      $(patsubst %.cpp, %$(CFG_OBJ_SUFFIX), $<), \
      $(RUNTIME_INCS)) $< $(CFG_PATH_MUNGE) >$@.tmp \
      && mv $@.tmp $@

%.d: %.ml $(MKFILES)
	@$(call CFG_ECHO, dep: $<)
	$(CFG_QUIET)ocamldep$(OPT) $(ML_DEP_INCS) $< $(CFG_PATH_MUNGE) >$@.tmp \
    && mv $@.tmp $@

%.d: %.mli $(MKFILES)
	@$(call CFG_ECHO, dep: $<)
	$(CFG_QUIET)ocamldep$(OPT) $(ML_DEP_INCS) $< $(CFG_PATH_MUNGE) >$@.tmp \
    && mv $@.tmp $@

ifneq ($(MAKECMDGOALS),clean)
-include $(ML_DEPFILES) $(C_DEPFILES)
endif

RUSTBOOT_PROBE := $(wildcard $(CFG_BOOT))

ifneq ($(RUSTBOOT_PROBE),)
CFG_INFO := $(info cfg: using built $(CFG_BOOT) for rust deps)
STDLIB_DEPFILE := $(CFG_STDLIB).d
CRATE_DEPFILES := $(ALL_TEST_CRATES:%.rc=%.d) $(STDLIB_DEPFILE)

$(STDLIB_DEPFILE): $(STDLIB_CRATE) $(MKFILES) $(CFG_BOOT)
	@$(call CFG_ECHO, dep: $<)
	$(BOOT) -shared -rdeps $< $(CFG_PATH_MUNGE) >$@.tmp \
    && mv $@.tmp $@

%.d: %.rc $(MKFILES) $(CFG_BOOT)
	@$(call CFG_ECHO, dep: $<)
	$(BOOT) -rdeps $< $(CFG_PATH_MUNGE) >$@.tmp \
    && mv $@.tmp $@

%.d: %.rs $(MKFILES) $(CFG_BOOT)
	@$(call CFG_ECHO, dep: $<)
	$(BOOT) -rdeps $< $(CFG_PATH_MUNGE) >$@.tmp \
    && mv $@.tmp $@

ifneq ($(MAKECMDGOALS),clean)
-include $(CRATE_DEPFILES)
endif
endif

######################################################################
# Distribution
######################################################################

PKG_NAME := rust
PKG_VER  := $(shell date +"%Y-%m-%d")-snap
PKG_DIR := $(PKG_NAME)-$(PKG_VER)
PKG_TAR := $(PKG_DIR).tar.gz
PKG_3RDPARTY := rt/valgrind.h rt/memcheck.h \
                rt/isaac/rand.h rt/isaac/standard.h \
                rt/uthash/uthash.h rt/uthash/utlist.h \
                rt/bigint/bigint.h rt/bigint/bigint_int.cpp \
                rt/bigint/bigint_ext.cpp rt/bigint/low_primes.h
PKG_FILES := README boot/README \
             $(wildcard etc/*.*) \
             $(MKFILES) $(BOOT_MLS) boot/fe/lexer.mll \
             $(COMPILER_INPUTS) \
             $(STDLIB_CRATE) $(STDLIB_INPUTS) \
             $(RUNTIME_CS) $(RUNTIME_HDR) $(PKG_3RDPARTY) \
             $(ALL_TEST_INPUTS)

dist:
	@$(call CFG_ECHO, making dist dir)
	$(CFG_QUIET)mkdir -p dist/$(PKG_DIR)
	$(CFG_QUIET)tar -c $(PKG_FILES) | tar -x -C dist/$(PKG_DIR)
	$(CFG_QUIET)cp ../LICENSE.txt dist/$(PKG_DIR)
	$(CFG_QUIET)tar -czf $(PKG_TAR) -C dist $(PKG_DIR)
	$(CFG_QUIET)rm -Rf dist

distcheck:
	@$(call CFG_ECHO, making dist dir)
	$(CFG_QUIET)rm -Rf $(PKG_NAME)-*.tar.gz dist
	$(CFG_QUIET)mkdir -p dist/$(PKG_DIR)
	$(CFG_QUIET)tar -c $(PKG_FILES) | tar -x -C dist/$(PKG_DIR)
	@$(call CFG_ECHO, making 'check' in dist dir)
	$(CFG_QUIET)make -C dist/$(PKG_DIR) check
	$(CFG_QUIET)make -C dist/$(PKG_DIR) clean
	@$(call CFG_ECHO, making tarball)
	$(CFG_QUIET)cp ../LICENSE.txt dist/$(PKG_DIR)
	$(CFG_QUIET)tar -czf $(PKG_TAR) -C dist $(PKG_DIR)
	$(CFG_QUIET)rm -Rf dist
	@echo
	@echo -----------------------------------------------
	@echo $(PKG_TAR) ready for distribution
	@echo -----------------------------------------------


######################################################################
# Cleanup
######################################################################

.PHONY: clean native-clean tidy

tidy:
	@$(call CFG_ECHO, check: formatting)
	$(CFG_QUIET) python etc/tidy.py \
      $(wildcard ../*.txt) \
      ../README \
      $(filter-out $(GENERATED) $(PKG_3RDPARTY) etc/%, $(PKG_FILES))

runtime-clean:
	$(CFG_QUIET)rm -f $(RUNTIME_OBJS) $(CFG_RUNTIME)

clean:
	@$(call CFG_ECHO, cleaning)
	$(CFG_QUIET)rm -f $(RUNTIME_OBJS)
	$(CFG_QUIET)rm -f $(BOOT_CMOS) $(BOOT_CMIS) $(BOOT_CMXS) $(BOOT_OBJS)
	$(CFG_QUIET)rm -f $(CFG_RUSTC)
	$(CFG_QUIET)rm -f $(ML_DEPFILES) $(C_DEPFILES) $(CRATE_DEPFILES)
	$(CFG_QUIET)rm -f $(ML_DEPFILES:%.d=%.d.tmp)
	$(CFG_QUIET)rm -f $(C_DEPFILES:%.d=%.d.tmp)
	$(CFG_QUIET)rm -f $(CRATE_DEPFILES:%.d=%.d.tmp)
	$(CFG_QUIET)rm -f $(GENERATED)
	$(CFG_QUIET)rm -f $(CFG_BOOT) $(CFG_RUNTIME) $(CFG_STDLIB)
	$(CFG_QUIET)rm -Rf $(PKG_NAME)-*.tar.gz dist
	$(CFG_QUIET)rm -f $(foreach ext,cmx cmi cmo cma o a d exe,\
                        $(wildcard boot/*/*.$(ext) boot/*/*/*.$(ext))) 
	$(CFG_QUIET)rm -Rf $(foreach ext,out llvm x86 o s exe dSYM,\
                        $(wildcard test/*/*.$(ext))) 


# Local Variables:
# mode: makefile-gmake
# fill-column: 78;
# buffer-file-coding-system: utf-8-unix
# compile-command: "make -k 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
# End:
