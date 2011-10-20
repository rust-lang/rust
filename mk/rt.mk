######################################################################
# Runtime (C++) library variables
######################################################################

RUNTIME_CS := rt/sync/timer.cpp \
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
              rt/rust_uv.cpp \
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
              rt/arch/i386/context.cpp \

RUNTIME_LL :=

RUNTIME_S := rt/arch/i386/_context.S \
             rt/arch/i386/ccall.S

RUNTIME_HDR := rt/globals.h \
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
               rt/arch/i386/context.h \

ifeq ($(CFG_WINDOWSY), 1)
  LIBUV_OSTYPE := win
  LIBUV_LIB := rt/libuv/Default/obj.target/src/rt/libuv/libuv.a
else ifeq ($(CFG_OSTYPE), apple-darwin)
  LIBUV_OSTYPE := mac
  LIBUV_LIB := rt/libuv/Default/libuv.a
else
  LIBUV_OSTYPE := unix
  LIBUV_LIB := rt/libuv/Default/obj.target/src/rt/libuv/libuv.a
endif

RUNTIME_DEF := rt/rustrt$(CFG_DEF_SUFFIX)
RUNTIME_INCS := -I $(S)src/rt/isaac -I $(S)src/rt/uthash \
                -I $(S)src/rt/arch/i386 -I $(S)src/rt/libuv/include
RUNTIME_OBJS := $(RUNTIME_CS:.cpp=.o) $(RUNTIME_LL:.ll=.o) $(RUNTIME_S:.S=.o)
RUNTIME_LIBS := $(LIBUV_LIB)

rt/%.o: rt/%.cpp $(MKFILES)
	@$(call E, compile: $@)
	$(Q)$(call CFG_COMPILE_C, $@, $(RUNTIME_INCS)) $<

rt/%.o: rt/%.S $(MKFILES)
	@$(call E, compile: $@)
	$(Q)$(call CFG_COMPILE_C, $@, $(RUNTIME_INCS)) $<

rt/%.o: rt/%.ll $(MKFILES)
	@$(call E, llc: $@)
	$(Q)$(LLC) -filetype=obj -relocation-model=pic -march=x86 -o $@ $<

rt/$(CFG_RUNTIME): $(RUNTIME_OBJS) $(MKFILES) $(RUNTIME_HDR) $(RUNTIME_DEF) $(RUNTIME_LIBS)
	@$(call E, link: $@)
	$(Q)$(call CFG_LINK_C,$@, $(RUNTIME_OBJS) \
	  $(CFG_GCCISH_POST_LIB_FLAGS) $(RUNTIME_LIBS) \
	  $(CFG_LIBUV_LINK_FLAGS),$(RUNTIME_DEF),$(CFG_RUNTIME))

# FIXME: For some reason libuv's makefiles can't figure out the correct definition
# of CC on the mingw I'm using, so we are explicitly using gcc. Also, we
# have to list environment variables first on windows... mysterious
$(LIBUV_LIB): $(wildcard \
                     $(S)src/rt/libuv/* \
                     $(S)src/rt/libuv/*/* \
                     $(S)src/rt/libuv/*/*/* \
                     $(S)src/rt/libuv/*/*/*/*)
	$(Q)$(MAKE) -C $(S)mk/libuv/$(LIBUV_OSTYPE) \
		CFLAGS="-m32" LDFLAGS="-m32" \
		CC="$(CFG_GCCISH_CROSS)$(CC)" \
		CXX="$(CFG_GCCISH_CROSS)$(CXX)" \
		AR="$(CFG_GCCISH_CROSS)$(AR)" \
		builddir_name="$(CFG_BUILD_DIR)/rt/libuv" \
		V=$(VERBOSE) FLOCK= uv

# These could go in rt.mk or rustllvm.mk, they're needed for both.

%.linux.def:    %.def.in $(MKFILES)
	@$(call E, def: $@)
	$(Q)echo "{" > $@
	$(Q)sed 's/.$$/&;/' $< >> $@
	$(Q)echo "};" >> $@

%.darwin.def:	%.def.in $(MKFILES)
	@$(call E, def: $@)
	$(Q)sed 's/^./_&/' $< > $@

ifdef CFG_WINDOWSY
%.def:	%.def.in $(MKFILES)
	@$(call E, def: $@)
	$(Q)echo LIBRARY $* > $@
	$(Q)echo EXPORTS >> $@
	$(Q)sed 's/^./    &/' $< >> $@
endif
