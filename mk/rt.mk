# This is a procedure to define the targets for building
# the runtime.
#
# Argument 1 is the target triple.
#
# This is not really the right place to explain this, but
# for those of you who are not Makefile gurus, let me briefly
# cover the $ expansion system in use here, because it
# confused me for a while!  The variable DEF_RUNTIME_TARGETS
# will be defined once and then expanded with different
# values substituted for $(1) each time it is called.
# That resulting text is then eval'd.
#
# For most variables, you could use a single $ sign.  The result
# is that the substitution would occur when the CALL occurs,
# I believe.  The problem is that the automatic variables $< and $@
# need to be expanded-per-rule.  Therefore, for those variables at
# least, you need $$< and $$@ in the variable text.  This way, after
# the CALL substitution occurs, you will have $< and $@.  This text
# will then be evaluated, and all will work as you like.
#
# Reader beware, this explanantion could be wrong, but it seems to
# fit the experimental data (i.e., I was able to get the system
# working under these assumptions).

# Hack for passing flags into LIBUV, see below.
LIBUV_FLAGS_i386 = -m32 -fPIC
LIBUV_FLAGS_x86_64 = -m64 -fPIC
ifeq ($(OSTYPE_$(1)), linux-androideabi)
LIBUV_FLAGS_arm = -fPIC -DANDROID -std=gnu99
else
LIBUV_FLAGS_arm = -fPIC -std=gnu99
endif
LIBUV_FLAGS_mips = -fPIC -mips32r2 -msoft-float -mabi=32

# when we're doing a snapshot build, we intentionally degrade as many
# features in libuv and the runtime as possible, to ease portability.

SNAP_DEFINES:=
ifneq ($(strip $(findstring snap,$(MAKECMDGOALS))),)
	SNAP_DEFINES=-DRUST_SNAPSHOT
endif

define DEF_RUNTIME_TARGETS

######################################################################
# Runtime (C++) library variables
######################################################################

# $(1) is the target triple
# $(2) is the stage number

RUNTIME_CFLAGS_$(1)_$(2) = -D_RUST_STAGE$(2)
RUNTIME_CXXFLAGS_$(1)_$(2) = -D_RUST_STAGE$(2)

RUNTIME_CXXS_$(1)_$(2) := \
              rt/sync/timer.cpp \
              rt/sync/lock_and_signal.cpp \
              rt/sync/rust_thread.cpp \
              rt/rust.cpp \
              rt/rust_builtin.cpp \
              rt/rust_run_program.cpp \
              rt/rust_env.cpp \
              rt/rust_rng.cpp \
              rt/rust_sched_loop.cpp \
              rt/rust_sched_launcher.cpp \
              rt/rust_sched_driver.cpp \
              rt/rust_scheduler.cpp \
              rt/rust_sched_reaper.cpp \
              rt/rust_task.cpp \
              rt/rust_stack.cpp \
              rt/rust_upcall.cpp \
              rt/rust_uv.cpp \
              rt/rust_crate_map.cpp \
              rt/rust_log.cpp \
              rt/rust_gc_metadata.cpp \
              rt/rust_util.cpp \
              rt/rust_exchange_alloc.cpp \
              rt/isaac/randport.cpp \
              rt/miniz.cpp \
              rt/rust_kernel.cpp \
              rt/rust_abi.cpp \
              rt/rust_debug.cpp \
              rt/memory_region.cpp \
              rt/boxed_region.cpp \
              rt/arch/$$(HOST_$(1))/context.cpp \
              rt/arch/$$(HOST_$(1))/gpr.cpp \
              rt/rust_android_dummy.cpp \
              rt/rust_test_helpers.cpp

RUNTIME_CS_$(1)_$(2) := rt/linenoise/linenoise.c rt/linenoise/utf8.c

RUNTIME_S_$(1)_$(2) := rt/arch/$$(HOST_$(1))/_context.S \
			rt/arch/$$(HOST_$(1))/ccall.S \
			rt/arch/$$(HOST_$(1))/record_sp.S

ifeq ($$(CFG_WINDOWSY_$(1)), 1)
  LIBUV_OSTYPE_$(1)_$(2) := win
  LIBUV_LIB_$(1)_$(2) := rt/$(1)/stage$(2)/libuv/libuv.a
else ifeq ($(OSTYPE_$(1)), apple-darwin)
  LIBUV_OSTYPE_$(1)_$(2) := mac
  LIBUV_LIB_$(1)_$(2) := rt/$(1)/stage$(2)/libuv/libuv.a
else ifeq ($(OSTYPE_$(1)), unknown-freebsd)
  LIBUV_OSTYPE_$(1)_$(2) := unix/freebsd
  LIBUV_LIB_$(1)_$(2) := rt/$(1)/stage$(2)/libuv/libuv.a
else ifeq ($(OSTYPE_$(1)), linux-androideabi)
  LIBUV_OSTYPE_$(1)_$(2) := unix/android
  LIBUV_LIB_$(1)_$(2) := rt/$(1)/stage$(2)/libuv/libuv.a
else
  LIBUV_OSTYPE_$(1)_$(2) := unix/linux
  LIBUV_LIB_$(1)_$(2) := rt/$(1)/stage$(2)/libuv/libuv.a
endif

RUNTIME_DEF_$(1)_$(2) := rt/rustrt$(CFG_DEF_SUFFIX_$(1))
RUNTIME_INCS_$(1)_$(2) := -I $$(S)src/rt -I $$(S)src/rt/isaac -I $$(S)src/rt/uthash \
                     -I $$(S)src/rt/arch/$$(HOST_$(1)) \
                     -I $$(S)src/rt/linenoise \
                     -I $$(S)src/libuv/include
RUNTIME_OBJS_$(1)_$(2) := $$(RUNTIME_CXXS_$(1)_$(2):rt/%.cpp=rt/$(1)/stage$(2)/%.o) \
                     $$(RUNTIME_CS_$(1)_$(2):rt/%.c=rt/$(1)/stage$(2)/%.o) \
                     $$(RUNTIME_S_$(1)_$(2):rt/%.S=rt/$(1)/stage$(2)/%.o)
ALL_OBJ_FILES += $$(RUNTIME_OBJS_$(1)_$(2))

MORESTACK_OBJ_$(1)_$(2) := rt/$(1)/stage$(2)/arch/$$(HOST_$(1))/morestack.o
ALL_OBJ_FILES += $$(MORESTACK_OBJS_$(1)_$(2))

RUNTIME_LIBS_$(1)_$(2) := $$(LIBUV_LIB_$(1)_$(2))

rt/$(1)/stage$(2)/%.o: rt/%.cpp $$(MKFILE_DEPS)
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_COMPILE_CXX_$(1), $$@, $$(RUNTIME_INCS_$(1)_$(2)) \
                 $$(SNAP_DEFINES) $$(RUNTIME_CXXFLAGS_$(1)_$(2))) $$<

rt/$(1)/stage$(2)/%.o: rt/%.c $$(MKFILE_DEPS)
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_COMPILE_C_$(1), $$@, $$(RUNTIME_INCS_$(1)_$(2)) \
                 $$(SNAP_DEFINES) $$(RUNTIME_CFLAGS_$(1)_$(2))) $$<

rt/$(1)/stage$(2)/%.o: rt/%.S  $$(MKFILE_DEPS) \
                     $$(LLVM_CONFIG_$$(CFG_BUILD_TRIPLE))
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_ASSEMBLE_$(1),$$@,$$<)

rt/$(1)/stage$(2)/arch/$$(HOST_$(1))/libmorestack.a: $$(MORESTACK_OBJ_$(1)_$(2))
	@$$(call E, link: $$@)
	$$(Q)$(AR_$(1)) rcs $$@ $$<

rt/$(1)/stage$(2)/$(CFG_RUNTIME_$(1)): $$(RUNTIME_OBJS_$(1)_$(2)) $$(MKFILE_DEPS) \
                        $$(RUNTIME_DEF_$(1)_$(2)) \
                        $$(RUNTIME_LIBS_$(1)_$(2))
	@$$(call E, link: $$@)
	$$(Q)$$(call CFG_LINK_CXX_$(1),$$@, $$(RUNTIME_OBJS_$(1)_$(2)) \
	  $$(CFG_GCCISH_POST_LIB_FLAGS_$(1)) $$(RUNTIME_LIBS_$(1)_$(2)) \
	  $$(CFG_LIBUV_LINK_FLAGS_$(1)),$$(RUNTIME_DEF_$(1)_$(2)),$$(CFG_RUNTIME_$(1)))

# FIXME: For some reason libuv's makefiles can't figure out the
# correct definition of CC on the mingw I'm using, so we are
# explicitly using gcc. Also, we have to list environment variables
# first on windows... mysterious

ifdef CFG_ENABLE_FAST_MAKE
LIBUV_DEPS := $$(S)/.gitmodules
else
LIBUV_DEPS := $$(wildcard \
              $$(S)src/libuv/* \
              $$(S)src/libuv/*/* \
              $$(S)src/libuv/*/*/* \
              $$(S)src/libuv/*/*/*/*)
endif

# XXX: Shouldn't need platform-specific conditions here
ifdef CFG_WINDOWSY_$(1)
$$(LIBUV_LIB_$(1)_$(2)): $$(LIBUV_DEPS)
	$$(Q)$$(MAKE) -C $$(S)src/libuv/ \
		builddir_name="$$(CFG_BUILD_DIR)/rt/$(1)/stage$(2)/libuv" \
		OS=mingw \
		V=$$(VERBOSE)
else ifeq ($(OSTYPE_$(1)), linux-androideabi)
$$(LIBUV_LIB_$(1)_$(2)): $$(LIBUV_DEPS)
	$$(Q)$$(MAKE) -C $$(S)src/libuv/ \
		CFLAGS="$$(CFG_GCCISH_CFLAGS) $$(LIBUV_FLAGS_$$(HOST_$(1))) $$(SNAP_DEFINES)" \
		LDFLAGS="$$(CFG_GCCISH_LINK_FLAGS) $$(LIBUV_FLAGS_$$(HOST_$(1)))" \
		CC="$$(CC_$(1))" \
		CXX="$$(CXX_$(1))" \
		AR="$$(AR_$(1))" \
		BUILDTYPE=Release \
		builddir_name="$$(CFG_BUILD_DIR)/rt/$(1)/stage$(2)/libuv" \
		host=android OS=linux \
		V=$$(VERBOSE)
else
$$(LIBUV_LIB_$(1)_$(2)): $$(LIBUV_DEPS)
	$$(Q)$$(MAKE) -C $$(S)src/libuv/ \
		CFLAGS="$$(CFG_GCCISH_CFLAGS) $$(LIBUV_FLAGS_$$(HOST_$(1))) $$(SNAP_DEFINES)" \
		LDFLAGS="$$(CFG_GCCISH_LINK_FLAGS) $$(LIBUV_FLAGS_$$(HOST_$(1)))" \
		CC="$$(CC_$(1))" \
		CXX="$$(CXX_$(1))" \
		AR="$$(AR_$(1))" \
		builddir_name="$$(CFG_BUILD_DIR)/rt/$(1)/stage$(2)/libuv" \
		V=$$(VERBOSE)
endif


# These could go in rt.mk or rustllvm.mk, they're needed for both.

# This regexp has a single $, escaped twice
%.bsd.def:    %.def.in $$(MKFILE_DEPS)
	@$$(call E, def: $$@)
	$$(Q)echo "{" > $$@
	$$(Q)sed 's/.$$$$/&;/' $$< >> $$@
	$$(Q)echo "};" >> $$@

%.linux.def:    %.def.in $$(MKFILE_DEPS)
	@$$(call E, def: $$@)
	$$(Q)echo "{" > $$@
	$$(Q)sed 's/.$$$$/&;/' $$< >> $$@
	$$(Q)echo "};" >> $$@

%.darwin.def:	%.def.in $$(MKFILE_DEPS)
	@$$(call E, def: $$@)
	$$(Q)sed 's/^./_&/' $$< > $$@

%.android.def:  %.def.in $$(MKFILE_DEPS)
	@$$(call E, def: $$@)
	$$(Q)echo "{" > $$@
	$$(Q)sed 's/.$$$$/&;/' $$< >> $$@
	$$(Q)echo "};" >> $$@

%.mingw32.def:	%.def.in $$(MKFILE_DEPS)
	@$$(call E, def: $$@)
	$$(Q)echo LIBRARY $$* > $$@
	$$(Q)echo EXPORTS >> $$@
	$$(Q)sed 's/^./    &/' $$< >> $$@

endef

# Instantiate template for all stages
$(foreach stage,$(STAGES), \
	$(foreach target,$(CFG_TARGET_TRIPLES), \
	 $(eval $(call DEF_RUNTIME_TARGETS,$(target),$(stage)))))
