######################################################################
# Doc variables and rules
######################################################################

DOCS :=


######################################################################
# Pandoc (reference-manual related)
######################################################################
ifeq ($(CFG_PANDOC),)
  $(info cfg: no pandoc found, omitting doc/rust.pdf)
else

DOCS += doc/rust.html doc/rust.css
doc/rust.html: rust.md doc/version.md doc/keywords.md
	@$(call E, pandoc: $@)
	$(Q)$(CFG_PANDOC) \
         --standalone --toc \
         --section-divs \
         --number-sections \
         --from=markdown --to=html \
         --css=rust.css \
         --output=$@ \
         $<
	@$(call E, cp: $(S)doc/rust.css)
	-$(Q)cp -a $(S)doc/rust.css doc/rust.css 2> /dev/null


  ifeq ($(CFG_PDFLATEX),)
    $(info cfg: no pdflatex found, omitting doc/rust.pdf)
  else
    ifeq ($(CFG_XETEX),)
      $(info cfg: no xetex found, disabling doc/rust.pdf)
    else
      ifeq ($(CFG_LUATEX),)
        $(info cfg: lacking luatex, disabling pdflatex)
      else

DOCS += doc/rust.pdf
doc/rust.tex: rust.md doc/version.md doc/keywords.md
	@$(call E, pandoc: $@)
	$(Q)$(CFG_PANDOC) \
         --standalone --toc \
         --number-sections \
         --from=markdown --to=latex \
         --output=$@ \
         $<

doc/rust.pdf: doc/rust.tex
	@$(call E, pdflatex: $@)
	$(Q)$(CFG_PDFLATEX) \
        -interaction=batchmode \
        -output-directory=doc \
        $<

      endif
    endif
  endif

######################################################################
# Node (tutorial related)
######################################################################
  ifeq ($(CFG_NODE),)
    $(info cfg: no node found, omitting doc/tutorial.html)
  else

DOCS += doc/tutorial.html
doc/tutorial.html: $(S)doc/tutorial.md
	@$(call E, cp: $(S)doc/rust.css)
	-$(Q)cp -a $(S)doc/rust.css doc/ 2> /dev/null
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
          $(CFG_PANDOC) --standalone --toc \
           --section-divs --number-sections \
           --from=markdown --to=html --css=rust.css \
           --output=$@

  endif
endif


######################################################################
# LLnextgen (grammar analysis from refman)
######################################################################
ifeq ($(CFG_LLNEXTGEN),)
  $(info cfg: no llnextgen found, omitting grammar-verification)
else
.PHONY: verify-grammar

doc/rust.g: rust.md $(S)src/etc/extract_grammar.py
	@$(call E, extract_grammar: $@)
	$(Q)$(S)src/etc/extract_grammar.py $< >$@

verify-grammar: doc/rust.g
	@$(call E, LLnextgen: $<)
	$(Q)$(CFG_LLNEXTGEN) --generate-lexer-wrapper=no $< >$@
	$(Q)rm -f doc/rust.c doc/rust.h
endif


######################################################################
# Naturaldocs (library reference related)
######################################################################

ifeq ($(CFG_NATURALDOCS),)
  $(info cfg: no naturaldocs found, omitting library doc build)
else

define libdoc
doc/$(1)/index.html: nd/$(1)/Languages.txt nd/$(1)/Topics.txt \
                     nd/$(1)/lib.css $(2)
	@$$(call E, naturaldocs: $$@)
	naturaldocs -i $(S)src/lib$(1) -o HTML doc/$(1) -p nd/$(1) -r -s Default lib

nd/$(1)/Languages.txt: $(S)doc/Languages.txt
	@$$(call E, cp: $$@)
	$(Q)cp $$< $$@

nd/$(1)/Topics.txt: $(S)doc/Topics.txt
	@$$(call E, cp: $$@)
	$(Q)cp $$< $$@

nd/$(1)/lib.css: $(S)doc/lib.css
	@$$(call E, cp: $$@)
	$(Q)cp $$< $$@

GENERATED += nd/$(1)/Languages.txt \
             nd/$(1)/Topics.txt \
             nd/$(1)/Menu.txt \
             nd/$(1)/Data

DOCS += doc/$(1)/index.html nd/$(1)/lib.css
endef

$(eval $(call libdoc,core,$(CORELIB_CRATE) $(CORELIB_INPUTS)))
$(eval $(call libdoc,std,$(STDLIB_CRATE) $(STDLIB_INPUTS)))
endif


ifdef CFG_DISABLE_DOCS
  $(info cfg: disabling doc build (CFG_DISABLE_DOCS))
  DOCS :=
endif


doc/version.md: $(MKFILE_DEPS) rust.md
	@$(call E, version-stamp: $@)
	$(Q)echo "$(CFG_VERSION)" >$@

doc/keywords.md: $(MKFILE_DEPS) rust.md
	@$(call E, grep -v: $$@)
	$(Q)grep -v '^#' $< >$@

GENERATED += doc/keywords.md doc/version.md

docs: $(DOCS)
