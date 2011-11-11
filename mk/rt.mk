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
LIBUV_FLAGS_i386 = -m32
LIBUV_FLAGS_x86_64 = -m64

define DEF_RUNTIME_TARGETS

######################################################################
# Runtime (C++) library variables
######################################################################

RUNTIME_CS_$(1) := \
              rt/sync/timer.cpp \
              rt/sync/sync.cpp \
              rt/sync/lock_and_signal.cpp \
              rt/rust.cpp \
              rt/rust_builtin.cpp \
              rt/rust_run_program.cpp \
              rt/rust_crate_cache.cpp \
              rt/rust_env.cpp \
              rt/rust_scheduler.cpp \
              rt/rust_task.cpp \
              rt/rust_task_list.cpp \
              rt/rust_chan.cpp \
              rt/rust_port.cpp \
              rt/rust_upcall.cpp \
              rt/rust_log.cpp \
              rt/rust_timer.cpp \
              rt/circular_buffer.cpp \
              rt/isaac/randport.cpp \
              rt/rust_srv.cpp \
              rt/rust_kernel.cpp \
              rt/rust_shape.cpp \
              rt/rust_obstack.cpp \
              rt/rust_gc.cpp \
              rt/rust_abi.cpp \
              rt/rust_cc.cpp \
              rt/rust_debug.cpp \
              rt/memory_region.cpp \
              rt/test/rust_test_harness.cpp \
              rt/test/rust_test_runtime.cpp \
              rt/test/rust_test_util.cpp \
              rt/arch/$$(HOST_$(1))/context.cpp

RUNTIME_S_$(1) := rt/arch/$$(HOST_$(1))/_context.S \
                  rt/arch/$$(HOST_$(1))/ccall.S \
                  rt/arch/$$(HOST_$(1))/morestack.S

RUNTIME_HDR_$(1) := rt/globals.h \
               rt/rust.h \
               rt/rust_abi.h \
               rt/rust_cc.h \
               rt/rust_debug.h \
               rt/rust_gc.h \
               rt/rust_internal.h \
               rt/rust_util.h \
               rt/rust_chan.h \
               rt/rust_env.h \
               rt/rust_obstack.h \
               rt/rust_unwind.h \
               rt/rust_upcall.h \
               rt/rust_port.h \
               rt/rust_scheduler.h \
               rt/rust_shape.h \
               rt/rust_task.h \
               rt/rust_task_list.h \
               rt/rust_log.h \
               rt/circular_buffer.h \
               rt/util/array_list.h \
               rt/util/indexed_list.h \
               rt/util/synchronized_indexed_list.h \
               rt/util/hash_map.h \
               rt/sync/sync.h \
               rt/sync/timer.h \
               rt/sync/lock_and_signal.h \
               rt/sync/lock_free_queue.h \
               rt/rust_srv.h \
               rt/rust_kernel.h \
               rt/memory_region.h \
               rt/memory.h \
               rt/test/rust_test_harness.h \
               rt/test/rust_test_runtime.h \
               rt/test/rust_test_util.h \
               rt/arch/$$(HOST_$(1))/context.h \
               rt/arch/$$(HOST_$(1))/regs.h

ifeq ($$(HOST_$(1)), i386)
  LIBUV_ARCH_$(1) := ia32
else
  LIBUV_ARCH_$(1) := x86_64
endif

ifeq ($$(CFG_WINDOWSY), 1)
  LIBUV_OSTYPE_$(1) := win
  LIBUV_LIB_$(1) := rt/$(1)/libuv/Default/obj.target/src/libuv/libuv.a
else ifeq ($(CFG_OSTYPE), apple-darwin)
  LIBUV_OSTYPE_$(1) := mac
  LIBUV_LIB_$(1) := rt/$(1)/libuv/Default/libuv.a
else
  LIBUV_OSTYPE_$(1) := unix
  LIBUV_LIB_$(1) := rt/$(1)/libuv/Default/obj.target/src/libuv/libuv.a
endif

RUNTIME_DEF_$(1) := rt/rustrt$$(CFG_DEF_SUFFIX)
RUNTIME_INCS_$(1) := -I $$(S)src/rt/isaac -I $$(S)src/rt/uthash \
                -I $$(S)src/rt/arch/$$(HOST_$(1)) \
				-I $$(S)src/libuv/include
RUNTIME_OBJS_$(1) := $$(RUNTIME_CS_$(1):rt/%.cpp=rt/$(1)/%.o) \
                     $$(RUNTIME_S_$(1):rt/%.S=rt/$(1)/%.o)
RUNTIME_LIBS_$(1) := $$(LIBUV_LIB_$(1))

rt/$(1)/%.o: rt/%.cpp $$(MKFILES)
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_COMPILE_C_$(1), $$@, $$(RUNTIME_INCS_$(1))) $$<

rt/$(1)/%.o: rt/%.S $$(MKFILES)
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_COMPILE_C_$(1), $$@, $$(RUNTIME_INCS_$(1))) $$<

rt/$(1)/$(CFG_RUNTIME): $$(RUNTIME_OBJS_$(1)) $$(MKFILES) \
			  				 $$(RUNTIME_HDR_$(1)) \
                             $$(RUNTIME_DEF_$(1)) \
                             $$(RUNTIME_LIBS_$(1))
	@$$(call E, link: $$@)
	$$(Q)$$(call CFG_LINK_C_$(1),$$@, $$(RUNTIME_OBJS_$(1)) \
	  $$(CFG_GCCISH_POST_LIB_FLAGS) $$(RUNTIME_LIBS_$(1)) \
	  $$(CFG_LIBUV_LINK_FLAGS),$$(RUNTIME_DEF_$(1)),$$(CFG_RUNTIME))

# FIXME: For some reason libuv's makefiles can't figure out the correct definition
# of CC on the mingw I'm using, so we are explicitly using gcc. Also, we
# have to list environment variables first on windows... mysterious
$$(LIBUV_LIB_$(1)): $$(wildcard \
                     $$(S)src/libuv/* \
                     $$(S)src/libuv/*/* \
                     $$(S)src/libuv/*/*/* \
                     $$(S)src/libuv/*/*/*/*)
	$$(Q)$$(MAKE) -C $$(S)mk/libuv/$$(LIBUV_ARCH_$(1))/$$(LIBUV_OSTYPE_$(1)) \
		CFLAGS="$$(LIBUV_FLAGS_$$(HOST_$(1)))" \
        LDFLAGS="$$(LIBUV_FLAGS_$$(HOST_$(1)))" \
		CC="$$(CFG_GCCISH_CROSS)$$(CC)" \
		CXX="$$(CFG_GCCISH_CROSS)$$(CXX)" \
		AR="$$(CFG_GCCISH_CROSS)$$(AR)" \
		builddir_name="$$(CFG_BUILD_DIR)/rt/$(1)/libuv" \
		V=$$(VERBOSE) FLOCK= uv

# These could go in rt.mk or rustllvm.mk, they're needed for both.

# This regexp has a single $, escaped twice
%.linux.def:    %.def.in $$(MKFILES)
	@$$(call E, def: $$@)
	$$(Q)echo "{" > $$@
	$$(Q)sed 's/.$$$$/&;/' $$< >> $$@
	$$(Q)echo "};" >> $$@

%.darwin.def:	%.def.in $$(MKFILES)
	@$$(call E, def: $$@)
	$$(Q)sed 's/^./_&/' $$< > $$@

ifdef CFG_WINDOWSY
%.def:	%.def.in $$(MKFILES)
	@$$(call E, def: $$@)
	$$(Q)echo LIBRARY $$* > $$@
	$$(Q)echo EXPORTS >> $$@
	$$(Q)sed 's/^./    &/' $$< >> $$@
endif

endef

# Instantiate template for all stages
$(foreach target,$(CFG_TARGET_TRIPLES), \
 $(eval $(call DEF_RUNTIME_TARGETS,$(target))))
