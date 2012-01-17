######################################################################
# Doc variables and rules
######################################################################

doc/version.md: $(MKFILE_DEPS) rust.md
	@$(call E, version-stamp: $@)
	$(Q)echo "$(CFG_VERSION)" >>$@

doc/keywords.md: $(MKFILE_DEPS) rust.md
	@$(call E, grep -v: $$@)
	$(Q)grep -v '^#' $< >$@

ifdef CFG_PANDOC

doc/rust.html: rust.md doc/version.md doc/keywords.md
	@$(call E, pandoc: $@)
	$(Q)$(CFG_PANDOC) \
         --standalone --toc \
         --section-divs \
         --number-sections \
         --from=markdown --to=html \
         --output=$@ \
         $<

ifdef CFG_PDFLATEX

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

ifdef CFG_NODE

doc/tutorial/web/index.html: \
        $(wildcard $(S)doc/tutorial/*.md)
	@$(call E, cp: $@)
	$(Q)cp -arv $(S)doc/tutorial doc/
	@$(call E, node: build.js)
	$(Q)cd doc/tutorial && $(CFG_NODE) build.js

endif

endif

ifdef CFG_LLNEXTGEN
doc/rust.g: rust.md $(S)src/etc/extract_grammar.py
	@$(call E, extract_grammar: $@)
	$(Q)$(S)src/etc/extract_grammar.py $< >$@

verify-grammar: doc/rust.g
	@$(call E, LLnextgen: $<)
	$(Q)$(CFG_LLNEXTGEN) --generate-lexer-wrapper=no $< >$@
	$(Q)rm -f doc/rust.c doc/rust.h
endif


GENERATED += doc/keywords.md doc/version.md

docsnap: doc/rust.pdf
	@$(call E, snap: doc/rust-$(shell date +"%Y-%m-%d")-snap.pdf)
	$(Q)mv $< doc/rust-$(shell date +"%Y-%m-%d")-snap.pdf

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

docs: $(DOCS)
