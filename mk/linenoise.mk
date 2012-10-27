######################################################################
# linenoise - minimalistic readline alternative used by the REPL
######################################################################

define DEF_LINENOISE_TARGETS

LINENOISE_CS_$(1) := $$(addprefix linenoise/, linenoise.c)
LINENOISE_OBJS_$(1) := $(LINENOISE_CS_$(1):linenoise/%.c=linenoise/$(1)/%.o)

ALL_OBJ_FILES += $$(LINENOISE_OBJS_$(1))

linenoise/$(1)/liblinenoise.a: $$(LINENOISE_OBJS_$(1))
	@$$(call E, link: $$@)
	$$(Q)ar rcs $$@ $$<

linenoise/$(1)/%.o: linenoise/%.c $$(MKFILE_DEPS)
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_COMPILE_C_$(1), $$@,) $$<
endef

# Instantiate template for all stages
$(foreach target,$(CFG_TARGET_TRIPLES), \
 $(eval $(call DEF_LINENOISE_TARGETS,$(target))))
