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

CFG_LLVM_CONFIG ?= llvm-config
CFG_BOOT_FLAGS := $(FLAGS)
CFG_RUSTC_FLAGS := -nowarn

# On Darwin, we need to run dsymutil so the debugging information ends
# up in the right place.  On other platforms, it automatically gets
# embedded into the executable, so use a no-op command.
DSYMUTIL := true

ifeq ($(CFG_OSTYPE), FreeBSD)
  CFG_RUNTIME := librustrt.so
  CFG_STDLIB := libstd.so
  CFG_GCC_CFLAGS += -fPIC -march=i686 -I/usr/local/include
  CFG_GCC_LINK_FLAGS += -shared -fPIC -lpthread -lrt
  ifeq ($(CFG_CPUTYPE), x86_64)
    CFG_GCC_CFLAGS += -m32
    CFG_GCC_LINK_FLAGS += -m32
  endif
  CFG_NATIVE := 1
  CFG_UNIXY := 1
  CFG_VALGRIND := $(shell which valgrind)
  ifdef CFG_VALGRIND
    CFG_VALGRIND += --leak-check=full \
                    --error-exitcode=1 \
                    --quiet --vex-iropt-level=0 \
                    --suppressions=etc/x86.supp
  endif
endif

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
                    --error-exitcode=1 \
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

ifdef CFG_LLVM_CONFIG
  LLC := "$(shell $(CFG_LLVM_CONFIG) --bindir)/llc"
  CFG_LLC_CFLAGS := -march=x86
  LLVM-DIS := "$(shell $(CFG_LLVM_CONFIG) --bindir)/llvm-dis"
endif

MKFILES := Makefile

# Add current directory to library search path of CFG_BOOT command line.

CFG_BOOT_FLAGS += -L .

# Commands we now know how to run.

BOOT := $(CFG_QUIET)OCAMLRUNPARAM="b1" $(CFG_BOOT) $(CFG_BOOT_FLAGS)
RUSTC := $(CFG_QUIET)$(CFG_RUSTC) $(CFG_RUSTC_FLAGS)

######################################################################
# Boot targets and rules
######################################################################

ML_DEP_INCS := -I boot/fe -I boot/me -I boot/be \
  -I boot/driver -I boot/util
ML_INCS := $(ML_DEP_INCS)
ML_LIBS := unix.cma nums.cma bigarray.cma
ML_NATIVE_LIBS := unix.cmxa nums.cmxa bigarray.cmxa
OCAMLC_FLAGS := -g $(ML_INCS) -w Ael -warn-error Ael
OCAMLOPT_FLAGS := $(ML_INCS) -w Ael -warn-error Ael \
                    $(CFG_OCAMLOPT_PROFILE_FLAGS)

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
DRIVER_TOP_MLS := $(addprefix boot/driver/, lib.ml glue.ml main.ml)

BOOT_MLS := $(UTIL_BOT_MLS) $(DRIVER_BOT_MLS) $(FE_MLS) $(IL_MLS) $(ME_MLS) \
  $(BE_MLS) $(DRIVER_TOP_MLS)
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
      --pretty=format:'let version = "prerelease (%h %ci)";;' >$@ || exit 1

loc:
	$(CFG_QUIET)wc -l $(BOOT_MLS) $(RUNTIME_CS) $(RUNTIME_HDR)

$(CFG_RUNTIME): $(RUNTIME_OBJS) $(MKFILES) $(RUNTIME_HDR)
	@$(call CFG_ECHO, compile: $<)
	$(CFG_QUIET)$(call CFG_LINK_C, $@) $(RUNTIME_OBJS)

$(CFG_STDLIB): $(STDLIB_CRATE) $(CFG_BOOT) $(MKFILES)
	@$(call CFG_ECHO, compile: $<)
	$(BOOT) -shared -o $@ $(STDLIB_CRATE)

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
	$(BOOT) -minimal -o $@ $<
	$(CFG_QUIET)chmod 0755 $@

self: $(CFG_RUSTC)
	@$(call CFG_ECHO, compile: $<)
	$(RUSTC) $(COMPILER_CRATE)


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

TEST_XFAILS_BOOT :=  $(TASK_XFAILS) \
                    $(NOMINAL_TAG_XFAILS) \
                    $(CONST_TAG_XFAILS) \
                    test/run-pass/arith-unsigned.rs \
                    test/run-pass/child-outlives-parent.rs \
                    test/run-pass/clone-with-exterior.rs \
                    test/run-pass/constrained-type.rs \
                    test/run-pass/destructor-ordering.rs \
                    test/run-pass/obj-as.rs \
                    test/run-pass/vec-slice.rs \
                    test/run-pass/fn-lval.rs \
                    test/run-pass/generic-fn-box.rs \
                    test/run-pass/generic-recursive-tag.rs \
                    test/run-pass/generic-tup.rs \
                    test/run-pass/iter-ret.rs \
                    test/run-pass/lib-io.rs \
                    test/run-pass/mlist-cycle.rs \
                    test/run-pass/obj-as.rs \
                    test/run-pass/task-comm.rs \
                    test/run-pass/task-comm-3.rs \
                    test/run-pass/vec-slice.rs \
                    test/run-pass/while-and-do-while.rs \
                    test/run-fail/task-comm-14.rs \
                    test/compile-fail/bad-expr-path.rs \
                    test/compile-fail/import.rs \
                    test/compile-fail/import2.rs \
                    test/compile-fail/import3.rs \
                    test/compile-fail/import4.rs \
                    test/compile-fail/bad-recv.rs \
                    test/compile-fail/bad-send.rs \
                    test/compile-fail/infinite-vec-type-recursion.rs \
                    test/compile-fail/tail-non-call.rs \
                    test/compile-fail/writing-through-read-alias.rs

TEST_XFAILS_RUSTC := $(CONST_TAG_XFAILS) \
                     $(addprefix test/run-pass/, \
                        acyclic-unwind.rs \
                        alt-pattern-drop.rs \
                        alt-type-simple.rs \
                        append-units.rs \
                        autoderef-objfn.rs \
                        basic-1.rs \
                        basic-2.rs \
                        basic.rs \
                        bind-obj-ctor.rs \
                        box-unbox.rs \
                        child-outlives-parent.rs \
                        clone-with-exterior.rs \
                        comm.rs \
                        constrained-type.rs \
                        destructor-ordering.rs \
                        drop-on-empty-block-exit.rs \
                        drop-parametric-closure-with-bound-box.rs \
                        export-non-interference.rs \
                        exterior.rs \
                        foreach-nested-2.rs \
                        foreach-nested.rs \
                        foreach-put-structured.rs \
                        foreach-simple-outer-slot.rs \
                        generic-fn-twice.rs \
                        generic-iter-frame.rs \
                        generic-recursive-tag.rs \
                        generic-tag-alt.rs \
                        generic-tag-values.rs \
                        integral-indexing.rs \
                        iter-range.rs \
                        iter-ret.rs \
                        lazychan.rs \
                        lib-bitv.rs \
                        lib-deque.rs \
                        lib-int.rs \
                        lib-io.rs \
                        lib-map.rs \
                        lib-rand.rs \
                        lib-sha1.rs \
                        lib-sort.rs \
                        lib-str.rs \
                        lib-task.rs \
                        lib-uint.rs \
                        lib-vec-str-conversions.rs \
                        lib-vec.rs \
                        list.rs \
                        many.rs \
                        mlist-cycle.rs \
                        mlist.rs \
                        mutable-alias-vec.rs \
                        native-opaque-type.rs \
                        obj-as.rs \
                        obj-dtor.rs \
                        obj-return-polytypes.rs \
                        pred.rs \
                        preempt.rs \
                        rt-circular-buffer.rs \
                        size-and-align.rs \
                        spawn-fn.rs \
                        spawn-module-qualified.rs \
                        spawn.rs \
                        str-append.rs \
                        str-concat.rs \
                        structured-compare.rs \
                        syntax-extension-fmt.rs \
                        syntax-extension-shell.rs \
                        task-comm-0.rs \
                        task-comm-1.rs \
                        task-comm-10.rs \
                        task-comm-11.rs \
                        task-comm-12.rs \
                        task-comm-13-thread.rs \
                        task-comm-13.rs \
                        task-comm-15.rs \
                        task-comm-2.rs \
                        task-comm-3.rs \
                        task-comm-4.rs \
                        task-comm-5.rs \
                        task-comm-6.rs \
                        task-comm-7.rs \
                        task-comm-8.rs \
                        task-comm-9.rs \
                        task-comm.rs \
                        task-killjoin.rs \
                        task-life-0.rs \
                        threads.rs \
                        type-sizes.rs \
                        use-import-export.rs \
                        user.rs \
                        utf8.rs \
                        vec-alloc-append.rs \
                        vec-append.rs \
                        vec-concat.rs \
                        vec-slice.rs \
                        while-prelude-drop.rs \
                        while-with-break.rs \
                        yield.rs \
                        yield2.rs \
                        multi.rc \
                        native-mod.rc \
                        native.rc \
                        ) \
                     $(filter-out \
                      $(addprefix test/compile-fail/, \
                        alt-tag-nullary.rs \
                        alt-tag-unary.rs \
                        arg-count-mismatch.rs \
                        arg-type-mismatch.rs \
                        bad-expr-path.rs \
                        bad-expr-path2.rs \
                        bogus-tag.rs \
                        fru-extra-field.rs \
                        import.rs \
                        import2.rs \
                        import3.rs \
                        import4.rs \
                        import-loop.rs \
                        import-loop-2.rs \
                        multiline-comment-line-tracking.rs \
                        output-type-mismatch.rs \
                        rec-missing-fields.rs \
                        reserved-dec.rs \
                        reserved-f128.rs \
                        reserved-f16.rs \
                        reserved-f80.rs \
                        reserved-m128.rs \
                        reserved-m32.rs \
                        reserved-m64.rs \
                        tail-non-call.rs \
                        tail-typeck.rs \
                        type-shadow.rs \
                        while-type-error.rs \
                        wrong-ret-type.rs \
                        ), \
                      $(wildcard test/*fail/*.rs test/*fail/*.rc))


ifdef MINGW_CROSS
TEST_XFAILS_BOOT += test/run-pass/native-mod.rc
TEST_XFAILS_RUSTC += test/run-pass/native-mod.rc
endif
ifdef CFG_WINDOWSY
TEST_XFAILS_BOOT += test/run-pass/native-mod.rc
TEST_XFAILS_RUSTC += test/run-pass/native-mod.rc
endif

RPASS_RC := $(wildcard test/run-pass/*.rc)
RPASS_RS := $(wildcard test/run-pass/*.rs)
RFAIL_RC := $(wildcard test/run-fail/*.rc)
RFAIL_RS := $(wildcard test/run-fail/*.rs)
CFAIL_RC := $(wildcard test/compile-fail/*.rc)
CFAIL_RS := $(wildcard test/compile-fail/*.rs)

ifdef CHECK_XFAILS
TEST_RPASS_CRATES_BOOT := $(filter $(TEST_XFAILS_BOOT), $(RPASS_RC))
TEST_RPASS_CRATES_RUSTC := $(filter $(TEST_XFAILS_RUSTC), $(RPASS_RC))
TEST_RPASS_SOURCES_BOOT := $(filter $(TEST_XFAILS_BOOT), $(RPASS_RS))
TEST_RPASS_SOURCES_RUSTC := $(filter $(TEST_XFAILS_RUSTC), $(RPASS_RS))
else
TEST_RPASS_CRATES_BOOT := $(filter-out $(TEST_XFAILS_BOOT), $(RPASS_RC))
TEST_RPASS_CRATES_RUSTC := $(filter-out $(TEST_XFAILS_RUSTC), $(RPASS_RC))
TEST_RPASS_SOURCES_BOOT := $(filter-out $(TEST_XFAILS_BOOT), $(RPASS_RS))
TEST_RPASS_SOURCES_RUSTC := $(filter-out $(TEST_XFAILS_RUSTC), $(RPASS_RS))
endif

TEST_RPASS_EXES_BOOT := \
              $(TEST_RPASS_CRATES_BOOT:.rc=.boot$(CFG_EXE_SUFFIX)) \
              $(TEST_RPASS_SOURCES_BOOT:.rs=.boot$(CFG_EXE_SUFFIX))
TEST_RPASS_EXES_RUSTC := \
              $(TEST_RPASS_CRATES_RUSTC:.rc=.rustc$(CFG_EXE_SUFFIX)) \
              $(TEST_RPASS_SOURCES_RUSTC:.rs=.rustc$(CFG_EXE_SUFFIX))

TEST_RPASS_OUTS_BOOT  := \
                 $(TEST_RPASS_EXES_BOOT:.boot$(CFG_EXE_SUFFIX)=.boot.out)
TEST_RPASS_OUTS_RUSTC := \
                 $(TEST_RPASS_EXES_RUSTC:.rustc$(CFG_EXE_SUFFIX)=.rustc.out)

TEST_RPASS_TMPS_BOOT  := \
  $(TEST_RPASS_EXES_BOOT:.boot$(CFG_EXE_SUFFIX)=.boot$(CFG_EXE_SUFFIX).tmp)
TEST_RPASS_TMPS_RUSTC := \
  $(TEST_RPASS_EXES_RUSTC:.rustc$(CFG_EXE_SUFFIX)=.rustc$(CFG_EXE_SUFFIX).tmp)


TEST_RFAIL_CRATES_BOOT := $(filter-out $(TEST_XFAILS_BOOT), $(RFAIL_RC))
TEST_RFAIL_CRATES_RUSTC := $(filter-out $(TEST_XFAILS_RUSTC), $(RFAIL_RC))
TEST_RFAIL_SOURCES_BOOT := $(filter-out $(TEST_XFAILS_BOOT), $(RFAIL_RS))
TEST_RFAIL_SOURCES_RUSTC := $(filter-out $(TEST_XFAILS_RUSTC), $(RFAIL_RS))

TEST_RFAIL_EXES_BOOT := \
              $(TEST_RFAIL_CRATES_BOOT:.rc=.boot$(CFG_EXE_SUFFIX)) \
              $(TEST_RFAIL_SOURCES_BOOT:.rs=.boot$(CFG_EXE_SUFFIX))
TEST_RFAIL_EXES_RUSTC := \
              $(TEST_RFAIL_CRATES_RUSTC:.rc=.rustc$(CFG_EXE_SUFFIX)) \
              $(TEST_RFAIL_SOURCES_RUSTC:.rs=.rustc$(CFG_EXE_SUFFIX))

TEST_RFAIL_OUTS_BOOT  := \
                  $(TEST_RFAIL_EXES_BOOT:.boot$(CFG_EXE_SUFFIX)=.boot.out)
TEST_RFAIL_OUTS_RUSTC := \
                  $(TEST_RFAIL_EXES_RUSTC:.rustc$(CFG_EXE_SUFFIX)=.rustc.out)

TEST_RFAIL_TMPS_BOOT  := \
  $(TEST_RFAIL_EXES_BOOT:.boot$(CFG_EXE_SUFFIX)=.boot$(CFG_EXE_SUFFIX).tmp)
TEST_RFAIL_TMPS_RUSTC := \
  $(TEST_RFAIL_EXES_RUSTC:.rustc$(CFG_EXE_SUFFIX)=.rustc$(CFG_EXE_SUFFIX).tmp)


TEST_CFAIL_CRATES_BOOT := $(filter-out $(TEST_XFAILS_BOOT), $(CFAIL_RC))
TEST_CFAIL_CRATES_RUSTC := $(filter-out $(TEST_XFAILS_RUSTC), $(CFAIL_RC))
TEST_CFAIL_SOURCES_BOOT := $(filter-out $(TEST_XFAILS_BOOT), $(CFAIL_RS))
TEST_CFAIL_SOURCES_RUSTC := $(filter-out $(TEST_XFAILS_RUSTC), $(CFAIL_RS))

TEST_CFAIL_EXES_BOOT := \
              $(TEST_CFAIL_CRATES_BOOT:.rc=.boot$(CFG_EXE_SUFFIX)) \
              $(TEST_CFAIL_SOURCES_BOOT:.rs=.boot$(CFG_EXE_SUFFIX))
TEST_CFAIL_EXES_RUSTC := \
              $(TEST_CFAIL_CRATES_RUSTC:.rc=.rustc$(CFG_EXE_SUFFIX)) \
              $(TEST_CFAIL_SOURCES_RUSTC:.rs=.rustc$(CFG_EXE_SUFFIX))

TEST_CFAIL_OUTS_BOOT := \
                $(TEST_CFAIL_EXES_BOOT:.boot$(CFG_EXE_SUFFIX)=.boot.out)
TEST_CFAIL_OUTS_RUSTC := \
                $(TEST_CFAIL_EXES_RUSTC:.rustc$(CFG_EXE_SUFFIX)=.rustc.out)

TEST_CFAIL_TMPS_BOOT := \
  $(TEST_CFAIL_EXES_BOOT:.boot$(CFG_EXE_SUFFIX)=.boot$(CFG_EXE_SUFFIX).tmp)
TEST_CFAIL_TMPS_RUSTC := \
  $(TEST_CFAIL_EXES_RUSTC:.rustc$(CFG_EXE_SUFFIX)=.rustc$(CFG_EXE_SUFFIX).tmp)


ALL_TEST_CRATES := $(TEST_CFAIL_CRATES_BOOT) \
                   $(TEST_RFAIL_CRATES_BOOT) \
                   $(TEST_RPASS_CRATES_BOOT)

ALL_TEST_SOURCES := $(TEST_CFAIL_SOURCES_BOOT) \
                    $(TEST_RFAIL_SOURCES_BOOT) \
                    $(TEST_RPASS_SOURCES_BOOT)

ALL_TEST_INPUTS := $(wildcard test/*/*.rs test/*/*/*.rs test/*/*.rc)

# The test suite currently relies on logging to validate results so
# make sure that logging uses the default configuration
unexport RUST_LOG


check_nocompile: $(TEST_CFAIL_OUTS_BOOT)

check: tidy \
       $(TEST_RPASS_EXES_BOOT) $(TEST_RFAIL_EXES_BOOT) \
       $(TEST_RPASS_OUTS_BOOT) $(TEST_RFAIL_OUTS_BOOT) \
       $(TEST_CFAIL_OUTS_BOOT)

compile-check: tidy \
       $(TEST_RPASS_EXES_BOOT) $(TEST_RFAIL_EXES_BOOT)


ifdef CFG_LLVM_CONFIG

ALL_TEST_CRATES += $(TEST_CFAIL_CRATES_RUSTC) \
                   $(TEST_RFAIL_CRATES_RUSTC) \
                   $(TEST_RPASS_CRATES_RUSTC)

ALL_TEST_SOURCES += $(TEST_CFAIL_SOURCES_RUSTC) \
                    $(TEST_RFAIL_SOURCES_RUSTC) \
                    $(TEST_RPASS_SOURCES_RUSTC)

check_nocompile: $(TEST_CFAIL_OUTS_RUSTC)

check:  tidy \
        $(TEST_RPASS_EXES_RUSTC) $(TEST_RFAIL_EXES_RUSTC) \
        $(TEST_RPASS_OUTS_RUSTC) $(TEST_RFAIL_OUTS_RUSTC) \
        $(TEST_CFAIL_OUTS_RUSTC)

compile-check: tidy \
       $(TEST_RPASS_EXES_RUSTC) $(TEST_RFAIL_EXES_RUSTC)

endif

BREQ := $(CFG_BOOT) $(CFG_RUNTIME) $(CFG_STDLIB)
SREQ := $(CFG_RUSTC) $(CFG_RUNTIME) $(CFG_STDLIB)

# Cancel the implicit .out rule in GNU make.
%.out: %

%.out: %.out.tmp
	$(CFG_QUIET)mv $< $@

test/run-pass/%.out.tmp: test/run-pass/%$(CFG_EXE_SUFFIX) $(CFG_RUNTIME)
	$(CFG_QUIET)rm -f $<.tmp
	@$(call CFG_ECHO, run: $<)
	$(CFG_QUIET)$(call CFG_RUN_TARG, $<) > $@

test/run-fail/%.out.tmp: test/run-fail/%$(CFG_EXE_SUFFIX) \
                                  $(CFG_RUNTIME)
	$(CFG_QUIET)rm -f $<.tmp
	@$(call CFG_ECHO, run: $<)
	$(CFG_QUIET)grep -q error-pattern test/run-fail/$(basename $*).rs
	$(CFG_QUIET)rm -f $@
	$(CFG_QUIET)$(call CFG_RUN_TARG, $<) >$@ 2>&1 ; X=$$? ; \
      if [ $$X -eq 0 ] ; then exit 1 ; else exit 0 ; fi
	$(CFG_QUIET)grep --text --quiet \
      "$$(grep error-pattern test/run-fail/$(basename $*).rs \
        | cut -d : -f 2- | tr -d '\n\r')" $@

test/compile-fail/%.boot.out.tmp: test/compile-fail/%.rs $(BREQ)
	@$(call CFG_ECHO, compile [boot]: $<)
	$(CFG_QUIET)grep -q error-pattern $<
	$(CFG_QUIET)rm -f $@
	$(BOOT) -o $(@:.out=$(CFG_EXE_SUFFIX)) $< >$@ 2>&1; test $$? -ne 0
	$(CFG_QUIET)grep --text --quiet \
      "$$(grep error-pattern $< | cut -d : -f 2- | tr -d '\n\r')" $@

test/compile-fail/%.rustc.out.tmp: test/compile-fail/%.rs $(SREQ)
	@$(call CFG_ECHO, compile [rustc]: $<)
	$(CFG_QUIET)grep -q error-pattern $<
	$(CFG_QUIET)rm -f $@
	$(RUSTC) -o $(@:.out=$(CFG_EXE_SUFFIX)) $< >$@ 2>&1; test $$? -ne 0
	$(CFG_QUIET)grep --text --quiet \
      "$$(grep error-pattern $< | cut -d : -f 2- | tr -d '\n\r')" $@

test/run-pass/%.boot$(CFG_EXE_SUFFIX): test/run-pass/%.rc $(BREQ)
	@$(call CFG_ECHO, compile [boot]: $<)
	$(BOOT) -o $@ $<

test/bench/shootout/%.boot$(CFG_EXE_SUFFIX): \
  test/bench/shootout/%.rs $(BREQ)
	@$(call CFG_ECHO, compile [boot]: $<)
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

%.rustc$(CFG_EXE_SUFFIX): %.o $(CFG_RUNTIME)
	@$(call CFG_ECHO, link [llvm]: $<)
	$(CFG_QUIET)gcc $(CFG_GCC_CFLAGS) -o $@ $< -L. -lrustrt
	@# dsymutil sometimes fails or prints a warning, but the
	@# program still runs.  Since it simplifies debugging other
	@# programs, I\'ll live with the noise.
	-$(CFG_QUIET)$(DSYMUTIL) $@

test/run-pass/%.bc: test/run-pass/%.rc $(SREQ)
	@$(call CFG_ECHO, compile [rustc]: $<)
	$(RUSTC) -o $@ $<

test/run-pass/%.boot$(CFG_EXE_SUFFIX): test/run-pass/%.rs $(BREQ)
	@$(call CFG_ECHO, compile [boot]: $<)
	$(BOOT) -o $@ $<

test/run-pass/%.bc: test/run-pass/%.rs $(SREQ)
	@$(call CFG_ECHO, compile [rustc]: $<)
	$(RUSTC) -o $@ $<

test/run-fail/%.boot$(CFG_EXE_SUFFIX): test/run-fail/%.rs $(BREQ)
	@$(call CFG_ECHO, compile [boot]: $<)
	$(BOOT) -o $@ $<

test/run-fail/%.bc: test/run-fail/%.rs $(SREQ)
	@$(call CFG_ECHO, compile [rustc]: $<)
	$(RUSTC) -o $@ $<


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
	$(CFG_QUIET)rm -Rf $(foreach ext,out llvm x86 boot rustc o s exe dSYM,\
                        $(wildcard test/*/*.$(ext)))


# Local Variables:
# mode: makefile-gmake
# fill-column: 78;
# buffer-file-coding-system: utf-8-unix
# compile-command: "make -k 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
# End:
