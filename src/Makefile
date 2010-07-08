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

CFG_LLVM_CONFIG := llvm-config
CFG_BOOT_FLAGS := $(FLAGS)

ifeq ($(CFG_OSTYPE), Linux)
  CFG_RUNTIME := librustrt.so
  CFG_STDLIB := libstd.so
  CFG_GCC_CFLAGS += -fPIC
  CFG_GCC_LINK_FLAGS += -shared -fPIC -ldl -lpthread
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
  CFG_GCC_CFLAGS += -m32
  CFG_GCC_LINK_FLAGS += -m32
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
  CFG_COMPILER := ./rustc.exe
  CFG_GCC_CFLAGS += -march=i686
  CFG_GCC_LINK_FLAGS += -shared -fPIC
  CFG_RUN_TARG = $(1)
  # FIXME: support msvc at some point
  CFG_GCC := 1
endif

ifdef CFG_UNIXY
  CFG_INFO := $(info cfg: unix-y environment)
  CFG_BOOT := ./rustboot
  CFG_COMPILER := ./rustc
  CFG_OBJ_SUFFIX := .o
  CFG_RUN_TARG = LD_LIBRARY_PATH=. $(CFG_VALGRIND) $(1)
  CFG_GCC := 1
  ifdef MINGW_CROSS
    CFG_INFO := $(info cfg: mingw-cross)
    CFG_GCC_CROSS := i586-mingw32msvc-
    CFG_BOOT_FLAGS += -t win32-x86-pe
    CFG_RUNTIME := rustrt.dll
    CFG_STDLIB := std.dll
    CFG_COMPILER := ./rustc.exe
    ifdef CFG_VALGRIND
      CFG_VALGRIND += wine
    endif
    CFG_OBJ_SUFFIX := .o
    CFG_EXE_SUFFIX := .exe
    CFG_GCC_CFLAGS :=
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

ifneq ($(CFG_LLVM_CONFIG),)
  CFG_LLVM_CONFIG := $(shell which $(CFG_LLVM_CONFIG))
endif
ifneq ($(CFG_LLVM_CONFIG),)
  CFG_LLVM_VERSION := $(shell $(CFG_LLVM_CONFIG) --version)
  ifeq ($(CFG_LLVM_VERSION),2.8svn)
    $(info cfg: using LLVM version 2.8svn)
  else
    CFG_LLVM_CONFIG :=
    $(info cfg: incompatible LLVM version $(CFG_LLVM_VERSION), \
      expected 2.8svn)
  endif
endif
ifdef CFG_LLVM_CONFIG
  VARIANT=llvm
  WHERE := $(shell ocamlc -where)
  LLVM_LIBS := llvm.cma llvm_bitwriter.cma
  LLVM_NATIVE_LIBS := llvm.cmxa llvm_bitwiter.cmxa
  LLVM_CLIBS := $(shell for c in `$(CFG_LLVM_CONFIG) --ldflags --libs` \
    -lllvm -lllvm_bitwriter; do echo -cclib && echo $$c; done | xargs echo)
  LLVM_INCS := -I boot/llvm -I $(WHERE)
  LLVM_MLS := $(addprefix boot/llvm/, llabi.ml llasm.ml llfinal.ml \
    lltrans.ml llemit.ml)
  CFG_LLC_CFLAGS := -march=x86
  $(info cfg: found llvm-config at $(CFG_LLVM_CONFIG))
else
  VARIANT=x86
  LLVM_CLIBS :=
  LLVM_INCS :=
  LLVM_MLS :=
  $(info cfg: disabling LLVM backend)
endif

MKFILES := Makefile

# add current directory to library search path of CFG_BOOT command line

CFG_BOOT_FLAGS += -L .

######################################################################
# Boot targets and rules
######################################################################

ML_INCS := -I boot/fe -I boot/me -I boot/be -I boot/driver/$(VARIANT) \
  -I boot/driver -I boot/util $(LLVM_INCS)
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
            type.ml dead.ml effect.ml typestate.ml loop.ml layout.ml  \
            transutil.ml trans.ml dwarf.ml)
FE_MLS := $(addprefix boot/fe/, ast.ml token.ml lexer.ml parser.ml pexp.ml \
            item.ml cexp.ml)
DRIVER_TOP_MLS := $(addprefix boot/driver/, lib.ml $(VARIANT)/glue.ml main.ml)

BOOT_MLS := $(UTIL_BOT_MLS) $(DRIVER_BOT_MLS) $(FE_MLS) $(IL_MLS) $(ME_MLS) \
  $(BE_MLS) $(LLVM_MLS) $(DRIVER_TOP_MLS)
BOOT_CMOS := $(BOOT_MLS:.ml=.cmo)
BOOT_CMXS := $(BOOT_MLS:.ml=.cmx)
BOOT_OBJS := $(BOOT_MLS:.ml=.o)
BOOT_CMIS := $(BOOT_MLS:.ml=.cmi)

RUNTIME_CS := rt/rust.cpp \
              rt/rust_builtin.cpp \
              rt/rust_crate.cpp \
              rt/rust_crate_cache.cpp \
              rt/rust_crate_reader.cpp \
              rt/rust_comm.cpp \
              rt/rust_dom.cpp \
              rt/rust_task.cpp \
              rt/rust_chan.cpp \
              rt/rust_upcall.cpp \
              rt/rust_log.cpp \
              rt/rust_timer.cpp \
              rt/isaac/randport.cpp

RUNTIME_HDR := rt/rust.h \
               rt/rust_dwarf.h \
               rt/rust_internal.h \
               rt/rust_util.h

RUNTIME_INCS := -Irt/isaac -Irt/uthash
RUNTIME_OBJS := $(RUNTIME_CS:.cpp=$(CFG_OBJ_SUFFIX))
RUNTIME_LIBS := $(CFG_RUNTIME_LIBS)

STDLIB_CRATE := lib/std.rc
STDLIB_INPUTS := $(wildcard lib/*.rc lib/*.rs lib/*/*.rs)
COMPILER_CRATE := comp/rustc.rc
COMPILER_INPUTS := $(wildcard comp/*.rc comp/*.rs comp/*/*.rs)

GENERATED := boot/fe/lexer.ml boot/util/version.ml

all: $(CFG_COMPILER) $(MKFILES) $(GENERATED)

boot/util/version.ml: Makefile
	$(CFG_QUIET)git log -1 \
      --format='let version = "prerelease (%h %ci)";;' >$@

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

$(CFG_COMPILER): $(COMPILER_INPUTS) $(CFG_BOOT) $(CFG_RUNTIME) $(CFG_STDLIB)
	@$(call CFG_ECHO, compile: $<)
	$(CFG_QUIET)OCAMLRUNPARAM="b1" $(CFG_BOOT) $(CFG_BOOT_FLAGS) -o $@ $<
	$(CFG_QUIET)chmod 0755 $@

self: $(CFG_COMPILER)
	@$(call CFG_ECHO, compile: $<)
	$(CFG_QUIET)$(call CFG_RUN_TARG, $(CFG_COMPILER)) $(COMPILER_CRATE)


######################################################################
# Testing
######################################################################

# There is some residual fallout from recent hacking on ty_box that should all
# start working more-or-less simultaneously when pwalton's work on type.ml
# appears. Until then, xfail a few cases where the typechecker mis-informs
# trans about TY_box / TY_mutable interactions.

MUT_BOX_XFAILS := $(addprefix test/run-pass/, \
                    deref.rs \
                    generic-exterior-box.rs \
                    output-slot-variants.rs \
                   )

TEST_XFAILS_X86 :=  $(MUT_BOX_XFAILS) \
                    test/run-pass/mlist-cycle.rs \
                    test/run-pass/clone-with-exterior.rs \
                    test/run-pass/obj-as.rs \
                    test/run-pass/rec-auto.rs \
                    test/run-pass/vec-slice.rs \
                    test/run-pass/fn-lval.rs \
                    test/run-pass/generic-fn-infer.rs \
                    test/run-pass/generic-fn-twice.rs \
                    test/run-pass/generic-recursive-tag.rs \
                    test/run-pass/mutable-vec-drop.rs \
                    test/run-pass/bind-obj-ctor.rs \
                    test/run-pass/task-comm.rs \
                    test/compile-fail/rec-missing-fields.rs \
                    test/compile-fail/bad-send.rs \
                    test/compile-fail/bad-recv.rs \
                    test/compile-fail/infinite-tag-type-recursion.rs \
                    test/compile-fail/infinite-vec-type-recursion.rs

TEST_XFAILS_LLVM := $(addprefix test/run-pass/, \
                      acyclic-unwind.rs \
                      alt-pattern-simple.rs \
                      alt-tag.rs \
                      argv.rs \
                      basic.rs \
                      bind-obj-ctor.rs \
                      bind-thunk.rs \
                      bind-trivial.rs \
                      bitwise.rs \
                      box-unbox.rs \
                      cast.rs \
                      char.rs \
                      clone-with-exterior.rs \
                      comm.rs \
                      command-line-args.rs \
                      complex.rs \
                      dead-code-one-arm-if.rs \
                      deep.rs \
                      deref.rs \
                      div-mod.rs \
                      drop-on-ret.rs \
                      else-if.rs \
                      export-non-interference.rs \
                      exterior.rs \
                      fn-lval.rs \
                      foreach-put-structured.rs \
                      foreach-simple-outer-slot.rs \
                      foreach-simple.rs \
                      fun-call-variants.rs \
                      fun-indirect-call.rs \
                      generic-derived-type.rs \
                      generic-drop-glue.rs \
                      generic-exterior-box.rs \
                      generic-fn-infer.rs \
                      generic-fn-twice.rs \
                      generic-fn.rs \
                      generic-obj-with-derived-type.rs \
                      generic-obj.rs \
                      generic-recursive-tag.rs \
                      generic-tag-alt.rs \
                      generic-tag.rs \
                      generic-type-synonym.rs \
                      generic-type.rs \
                      i32-sub.rs \
                      i8-incr.rs \
                      import.rs \
                      inner-module.rs \
                      item-name-overload.rs \
                      large-records.rs \
                      lazy-and-or.rs \
                      lazychan.rs \
                      linear-for-loop.rs \
                      list.rs \
                      many.rs \
                      mlist-cycle.rs \
                      mlist.rs \
                      mutable-alias-vec.rs \
                      mutable-vec-drop.rs \
                      mutual-recursion-group.rs \
                      native-mod.rc \
                      native-opaque-type.rs \
                      native.rc \
                      obj-as.rs \
                      obj-drop.rs \
                      obj-dtor.rs \
                      obj-recursion.rs \
                      obj-with-vec.rs \
                      opeq.rs \
                      output-slot-variants.rs \
                      preempt.rs \
                      pred.rs \
                      readalias.rs \
                      rec-auto.rs \
                      rec-extend.rs \
                      rec-tup.rs \
                      rec.rs \
                      return-nil.rs \
                      simple-obj.rs \
                      spawn-fn.rs \
                      spawn.rs \
                      stateful-obj.rs \
                      str-append.rs \
                      str-concat.rs \
                      str-idx.rs \
                      syntax-extension.rs \
                      tag.rs \
                      tail-cps.rs \
                      tail-direct.rs \
                      threads.rs \
                      tup.rs \
                      type-sizes.rs \
                      u32-decr.rs \
                      u8-incr-decr.rs \
                      u8-incr.rs \
                      unit.rs \
                      user.rs \
                      use-uninit.rs \
                      utf8.rs \
                      vec-append.rs \
                      vec-concat.rs \
                      vec-drop.rs \
                      vec-slice.rs \
                      vec.rs \
                      writealias.rs \
                      yield.rs \
                      yield2.rs \
                      task-comm.rs \
                     ) \
                    $(addprefix test/run-fail/, \
                      explicit-fail.rs \
                      fail.rs \
                      linked-failure.rs \
                      pred.rs \
                      vec-overrun.rs \
                      str-overrun.rs \
                      vec-underrun.rs \
                     ) \
                    $(addprefix test/compile-fail/, \
                      rec-missing-fields.rs \
                      bad-send.rs \
                      bad-recv.rs \
                      infinite-tag-type-recursion.rs \
                      infinite-vec-type-recursion.rs \
                     )

ifdef MINGW_CROSS
TEST_XFAILS_X86 += test/run-pass/native-mod.rc
TEST_XFAILS_LLVM += test/run-pass/native-mod.rc
endif
ifdef CFG_WINDOWSY
TEST_XFAILS_X86 += test/run-pass/native-mod.rc
TEST_XFAILS_LLVM += test/run-pass/native-mod.rc
endif

RPASS_RC := $(wildcard test/run-pass/*.rc)
RPASS_RS := $(wildcard test/run-pass/*.rs)
RFAIL_RC := $(wildcard test/run-fail/*.rc)
RFAIL_RS := $(wildcard test/run-fail/*.rs)
CFAIL_RC := $(wildcard test/compile-fail/*.rc)
CFAIL_RS := $(wildcard test/compile-fail/*.rs)

TEST_RPASS_CRATES_X86 := $(filter-out $(TEST_XFAILS_X86), $(RPASS_RC))
TEST_RPASS_CRATES_LLVM := $(filter-out $(TEST_XFAILS_LLVM), $(RPASS_RC))
TEST_RPASS_SOURCES_X86 := $(filter-out $(TEST_XFAILS_X86), $(RPASS_RS))
TEST_RPASS_SOURCES_LLVM := $(filter-out $(TEST_XFAILS_LLVM), $(RPASS_RS))
TEST_RPASS_EXTRAS := $(wildcard test/run-pass/*/*.rs)
TEST_RPASS_EXES_X86 := \
                      $(TEST_RPASS_CRATES_X86:.rc=.x86$(CFG_EXE_SUFFIX)) \
                      $(TEST_RPASS_SOURCES_X86:.rs=.x86$(CFG_EXE_SUFFIX))
TEST_RPASS_EXES_LLVM := \
                      $(TEST_RPASS_CRATES_LLVM:.rc=.llvm$(CFG_EXE_SUFFIX)) \
                      $(TEST_RPASS_SOURCES_LLVM:.rs=.llvm$(CFG_EXE_SUFFIX))
TEST_RPASS_OUTS_X86  := \
                    $(TEST_RPASS_EXES_X86:.x86$(CFG_EXE_SUFFIX)=.x86.out)
TEST_RPASS_OUTS_LLVM := \
                    $(TEST_RPASS_EXES_LLVM:.llvm$(CFG_EXE_SUFFIX)=.llvm.out)


TEST_RFAIL_CRATES_X86 := $(filter-out $(TEST_XFAILS_X86), $(RFAIL_RC))
TEST_RFAIL_CRATES_LLVM := $(filter-out $(TEST_XFAILS_LLVM), $(RFAIL_RC))
TEST_RFAIL_SOURCES_X86 := $(filter-out $(TEST_XFAILS_X86), $(RFAIL_RS))
TEST_RFAIL_SOURCES_LLVM := $(filter-out $(TEST_XFAILS_LLVM), $(RFAIL_RS))
TEST_RFAIL_EXTRAS := $(wildcard test/run-fail/*/*.rs)
TEST_RFAIL_EXES_X86 := \
                      $(TEST_RFAIL_CRATES_X86:.rc=.x86$(CFG_EXE_SUFFIX)) \
                      $(TEST_RFAIL_SOURCES_X86:.rs=.x86$(CFG_EXE_SUFFIX))
TEST_RFAIL_EXES_LLVM := \
                      $(TEST_RFAIL_CRATES_LLVM:.rc=.llvm$(CFG_EXE_SUFFIX)) \
                      $(TEST_RFAIL_SOURCES_LLVM:.rs=.llvm$(CFG_EXE_SUFFIX))
TEST_RFAIL_OUTS_X86  := \
                    $(TEST_RFAIL_EXES_X86:.x86$(CFG_EXE_SUFFIX)=.x86.out)
TEST_RFAIL_OUTS_LLVM := \
                    $(TEST_RFAIL_EXES_LLVM:.llvm$(CFG_EXE_SUFFIX)=.llvm.out)


TEST_CFAIL_CRATES_X86 := $(filter-out $(TEST_XFAILS_X86), $(CFAIL_RC))
TEST_CFAIL_CRATES_LLVM := $(filter-out $(TEST_XFAILS_LLVM), $(CFAIL_RC))
TEST_CFAIL_SOURCES_X86 := $(filter-out $(TEST_XFAILS_X86), $(CFAIL_RS))
TEST_CFAIL_SOURCES_LLVM := $(filter-out $(TEST_XFAILS_LLVM), $(CFAIL_RS))
TEST_CFAIL_EXTRAS := $(wildcard test/compile-fail/*/*.rs)
TEST_CFAIL_EXES_X86 := \
                  $(TEST_CFAIL_CRATES_X86:.rc=.x86$(CFG_EXE_SUFFIX)) \
                  $(TEST_CFAIL_SOURCES_X86:.rs=.x86$(CFG_EXE_SUFFIX))
TEST_CFAIL_EXES_LLVM := \
                  $(TEST_CFAIL_CRATES_LLVM:.rc=.llvm$(CFG_EXE_SUFFIX)) \
                  $(TEST_CFAIL_SOURCES_LLVM:.rs=.llvm$(CFG_EXE_SUFFIX))
TEST_CFAIL_OUTS_X86 := \
                $(TEST_CFAIL_EXES_X86:.x86$(CFG_EXE_SUFFIX)=.x86.out)
TEST_CFAIL_OUTS_LLVM := \
                $(TEST_CFAIL_EXES_LLVM:.llvm$(CFG_EXE_SUFFIX)=.llvm.out)

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


ifeq ($(VARIANT),llvm)
ALL_TEST_CRATES += $(TEST_CFAIL_CRATES_LLVM) \
                   $(TEST_RFAIL_CRATES_LLVM) \
                   $(TEST_RPASS_CRATES_LLVM)

ALL_TEST_SOURCES += $(TEST_CFAIL_SOURCES_LLVM) \
                    $(TEST_RFAIL_SOURCES_LLVM) \
                    $(TEST_RPASS_SOURCES_LLVM)

check_nocompile: $(TEST_CFAIL_OUTS_LLVM)

check:  tidy \
        $(TEST_RPASS_EXES_LLVM) $(TEST_RFAIL_EXES_LLVM) \
        $(TEST_RPASS_OUTS_LLVM) $(TEST_RFAIL_OUTS_LLVM) \
        $(TEST_CFAIL_OUTS_LLVM)
endif

REQ := $(CFG_BOOT) $(CFG_RUNTIME) $(CFG_STDLIB)
BOOT := $(CFG_QUIET)OCAMLRUNPARAM="b1" $(CFG_BOOT) $(CFG_BOOT_FLAGS)

# Cancel the implicit .out rule in GNU make.
%.out: %

%.out: %.out.tmp
	$(CFG_QUIET)mv $< $@

test/run-pass/%.out.tmp: test/run-pass/%$(CFG_EXE_SUFFIX) $(CFG_RUNTIME)
	@$(call CFG_ECHO, run: $<)
	$(CFG_QUIET)$(call CFG_RUN_TARG, $<) > $@

test/run-fail/%.out.tmp: test/run-fail/%$(CFG_EXE_SUFFIX) $(CFG_RUNTIME)
	@$(call CFG_ECHO, run: $<)
	$(CFG_QUIET)rm -f $@
	$(CFG_QUIET)$(call CFG_RUN_TARG, $<) >$@ 2>&1 ; X=$$? ; \
      if [ $$X -eq 0 ] ; then exit 1 ; else exit 0 ; fi
	$(CFG_QUIET)grep --text --quiet \
      "`awk -F: '/error-pattern/ { print $$2 }' \
        $(basename $(basename $(basename $@))).rs | tr -d '\n\r'`" $@

test/compile-fail/%.x86.out.tmp: test/compile-fail/%.rs $(REQ)
	@$(call CFG_ECHO, compile [x86]: $<)
	$(CFG_QUIET)rm -f $@
	$(BOOT) -o $(@:.out=$(CFG_EXE_SUFFIX)) $< >$@ 2>&1 || true
	$(CFG_QUIET)grep --text --quiet \
      "`awk -F: '/error-pattern/ { print $$2 }' $< | tr -d '\n\r'`" $@

test/compile-fail/%.llvm.out.tmp: test/compile-fail/%.rs $(REQ)
	@$(call CFG_ECHO, compile [llvm]: $<)
	$(CFG_QUIET)rm -f $@
	$(BOOT) -o $(@:.out=$(CFG_EXE_SUFFIX)) $< >$@ 2>&1 || true
	$(CFG_QUIET)grep --text --quiet \
      "`awk -F: '/error-pattern/ { print $$2 }' $< | tr -d '\n\r'`" $@

test/run-pass/%.x86$(CFG_EXE_SUFFIX): test/run-pass/%.rc $(REQ)
	@$(call CFG_ECHO, compile [x86]: $<)
	$(BOOT) -o $@ $<

%.s: %.bc
	@$(call CFG_ECHO, compile [llvm]: $<)
	$(CFG_QUIET)llc $(CFG_LLC_CFLAGS) -o $@ $<

%.llvm$(CFG_EXE_SUFFIX): %.s $(CFG_RUNTIME)
	@$(call CFG_ECHO, compile [llvm]: $<)
	$(CFG_QUIET)gcc $(CFG_GCC_CFLAGS) -o $@ $< -L. -lrustrt

test/run-pass/%.bc: test/run-pass/%.rc $(REQ)
	@$(call CFG_ECHO, compile [llvm]: $<)
	$(BOOT) -o $@ -llvm $<

test/run-pass/%.x86$(CFG_EXE_SUFFIX): test/run-pass/%.rs $(REQ)
	@$(call CFG_ECHO, compile [x86]: $<)
	$(BOOT) -o $@ $<

test/run-pass/%.bc: test/run-pass/%.rs $(REQ)
	@$(call CFG_ECHO, compile [llvm]: $<)
	$(BOOT) -o $@ -llvm $<

test/run-fail/%.x86$(CFG_EXE_SUFFIX): test/run-fail/%.rs $(REQ)
	@$(call CFG_ECHO, compile [x86]: $<)
	$(BOOT) -o $@ $<

test/run-fail/%.bc: test/run-fail/%.rs $(REQ)
	@$(call CFG_ECHO, compile [llvm]: $<)
	$(BOOT) -o $@ -llvm $<


######################################################################
# Auto-dependency
######################################################################

ML_DEPFILES := $(BOOT_MLS:%.ml=%.d)
C_DEPFILES := $(RUNTIME_CS:%.cpp=%.d)

%.d: %.cpp $(MKFILES)
	@$(call CFG_ECHO, dep: $<)
	$(CFG_QUIET)$(call CFG_DEPEND_C, $@ \
      $(patsubst %.cpp, %$(CFG_OBJ_SUFFIX), $<), \
      $(RUNTIME_INCS)) $< $(CFG_PATH_MUNGE) >$@

%.d: %.ml $(MKFILES)
	@$(call CFG_ECHO, dep: $<)
	$(CFG_QUIET)ocamldep$(OPT) $(ML_INCS) $< $(CFG_PATH_MUNGE) >$@

%.d: %.mli $(MKFILES)
	@$(call CFG_ECHO, dep: $<)
	$(CFG_QUIET)ocamldep$(OPT) $(ML_INCS) $< $(CFG_PATH_MUNGE) >$@

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
	$(BOOT) -shared -rdeps $< $(CFG_PATH_MUNGE) >$@

%.d: %.rc $(MKFILES) $(CFG_BOOT)
	@$(call CFG_ECHO, dep: $<)
	$(BOOT) -rdeps $< $(CFG_PATH_MUNGE) >$@

%.d: %.rs $(MKFILES) $(CFG_BOOT)
	@$(call CFG_ECHO, dep: $<)
	$(BOOT) -rdeps $< $(CFG_PATH_MUNGE) >$@

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
PKG_FILES := README \
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
      $(filter-out $(GENERATED) $(PKG_3RDPARTY), $(PKG_FILES))

runtime-clean:
	$(CFG_QUIET)rm -f $(RUNTIME_OBJS) $(CFG_RUNTIME)

clean:
	@$(call CFG_ECHO, cleaning)
	$(CFG_QUIET)rm -f $(RUNTIME_OBJS)
	$(CFG_QUIET)rm -f $(BOOT_CMOS) $(BOOT_CMIS) $(BOOT_CMXS) $(BOOT_OBJS)
	$(CFG_QUIET)rm -f $(CFG_COMPILER)
	$(CFG_QUIET)rm -f $(ML_DEPFILES) $(C_DEPFILES) $(CRATE_DEPFILES)
	$(CFG_QUIET)rm -f $(GENERATED)
	$(CFG_QUIET)rm -f $(CFG_BOOT) $(CFG_RUNTIME) $(CFG_STDLIB)
	$(CFG_QUIET)rm -f $(TEST_RPASS_EXES_X86) $(TEST_RPASS_OUTS_X86)
	$(CFG_QUIET)rm -f $(TEST_RPASS_EXES_LLVM) $(TEST_RPASS_OUTS_LLVM)
	$(CFG_QUIET)rm -f $(TEST_RFAIL_EXES_X86) $(TEST_RFAIL_OUTS_X86)
	$(CFG_QUIET)rm -f $(TEST_RFAIL_EXES_LLVM) $(TEST_RFAIL_OUTS_LLVM)
	$(CFG_QUIET)rm -f $(TEST_CFAIL_EXES_X86) $(TEST_CFAIL_OUTS_X86)
	$(CFG_QUIET)rm -f $(TEST_CFAIL_EXES_LLVM) $(TEST_CFAIL_OUTS_LLVM)
	$(CFG_QUIET)rm -Rf $(PKG_NAME)-*.tar.gz dist
	$(CFG_QUIET)rm -f $(foreach ext,cmx cmi cmo cma o a d exe,\
                        $(wildcard boot/*/*.$(ext) boot/*/*/*.$(ext)))
