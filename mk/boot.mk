######################################################################
# Bootstrap compiler variables and rules
######################################################################

ifdef CFG_BOOT_PROFILE
  $(info cfg: forcing native bootstrap compiler (CFG_BOOT_PROFILE))
  CFG_BOOT_NATIVE := 1
  CFG_OCAMLOPT_PROFILE_FLAGS := -p
endif

ifdef CFG_BOOT_DEBUG
  $(info cfg: forcing bytecode bootstrap compiler (CFG_BOOT_DEBUG))
  CFG_BOOT_NATIVE :=
endif

ifdef CFG_BOOT_NATIVE
  $(info cfg: building native bootstrap compiler)
else
  $(info cfg: building bytecode bootstrap compiler)
endif

GENERATED := boot/fe/lexer.ml boot/version.ml


# We must list them in link order.
# Nobody calculates the link-order DAG automatically, sadly.

BOOT_MLS :=                                              \
    $(addsuffix .ml,                                     \
        boot/version                                     \
        $(addprefix boot/util/, fmt common bits)         \
        $(addprefix boot/driver/, session)               \
        $(addprefix boot/fe/, ast token lexer parser     \
          extfmt pexp item cexp fuzz)                    \
        $(addprefix boot/be/, asm il abi)                \
        $(addprefix boot/me/, walk semant resolve alias  \
          simplify type dead layer typestate             \
         loop layout transutil trans dwarf)              \
        $(addprefix boot/be/, x86 ra pe elf macho)       \
        $(addprefix boot/driver/, lib glue main))        \

BOOT_CMOS := $(BOOT_MLS:.ml=.cmo)
BOOT_CMXS := $(BOOT_MLS:.ml=.cmx)
BOOT_OBJS := $(BOOT_MLS:.ml=.o)
BOOT_CMIS := $(BOOT_MLS:.ml=.cmi)

BS := $(S)src/boot

BOOT_ML_DEP_INCS := -I $(BS)/fe   -I $(BS)/me      \
                    -I $(BS)/be   -I $(BS)/driver  \
                    -I $(BS)/util -I boot

BOOT_ML_INCS    :=  -I boot/fe   -I boot/me      \
                    -I boot/be   -I boot/driver  \
                    -I boot/util -I boot

BOOT_ML_LIBS        := unix.cma  nums.cma  bigarray.cma
BOOT_ML_NATIVE_LIBS := unix.cmxa nums.cmxa bigarray.cmxa
BOOT_OCAMLC_FLAGS   := -g $(BOOT_ML_INCS) -w Ael -warn-error Ael
BOOT_OCAMLOPT_FLAGS := -g $(BOOT_ML_INCS) -w Ael -warn-error Ael

ifdef CFG_FLEXLINK
  BOOT_OCAMLOPT_FLAGS += -cclib -L/usr/lib
endif

BOOT := $(Q)OCAMLRUNPARAM="b1" boot/rustboot$(X) $(CFG_BOOT_FLAGS) -L stage0


ifdef CFG_BOOT_NATIVE
boot/rustboot$(X): $(BOOT_CMXS) $(MKFILES)
	@$(call E, link: $@)
	$(Q)ocamlopt$(OPT) -o $@ $(BOOT_OCAMLOPT_FLAGS) $(BOOT_ML_NATIVE_LIBS) \
        $(BOOT_CMXS)
else
boot/rustboot$(X): $(BOOT_CMOS) $(MKFILES)
	@$(call E, link: $@)
	$(Q)ocamlc$(OPT) -o $@ $(BOOT_OCAMLC_FLAGS) $(BOOT_ML_LIBS) $(BOOT_CMOS)
endif

boot/version.ml: $(MKFILES)
	@$(call E, git: $@)
	$(Q)(cd $(S) && git log -1 \
      --pretty=format:'let version = "prerelease (%h %ci)";;') >$@ || exit 1

%.cmo: %.ml $(MKFILES)
	@$(call E, compile: $@)
	$(Q)ocamlc$(OPT) -c -o $@ $(BOOT_OCAMLC_FLAGS) $<

%.cmo: %.cmi $(MKFILES)

%.cmx %.o: %.ml $(MKFILES)
	@$(call E, compile: $@)
	$(Q)ocamlopt$(OPT) -c -o $@ $(BOOT_OCAMLOPT_FLAGS) $<

%.ml: %.mll $(MKFILES)
	@$(call E, lex-gen: $@)
	$(Q)ocamllex$(OPT) -q -o $@ $<

