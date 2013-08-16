# Copyright 2012 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

######################################################################
# Doc variables and rules
######################################################################

DOCS :=
DOCS_L10N :=


######################################################################
# Docs, from pandoc, rustdoc (which runs pandoc), and node
######################################################################

doc/rust.css: rust.css
	@$(call E, cp: $@)
	$(Q)cp -a $< $@ 2> /dev/null

doc/manual.css: manual.css
	@$(call E, cp: $@)
	$(Q)cp -a $< $@ 2> /dev/null

ifeq ($(CFG_PANDOC),)
  $(info cfg: no pandoc found, omitting docs)
  NO_DOCS = 1
endif

ifeq ($(CFG_NODE),)
  $(info cfg: no node found, omitting docs)
  NO_DOCS = 1
endif

ifneq ($(NO_DOCS),1)

DOCS += doc/rust.html
doc/rust.html: rust.md doc/version_info.html doc/rust.css doc/manual.css
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
	"$(CFG_PANDOC)" \
         --standalone --toc \
         --section-divs \
         --number-sections \
         --from=markdown --to=html \
         --css=rust.css \
         --css=manual.css \
	     --include-before-body=doc/version_info.html \
         --output=$@

DOCS += doc/rust.tex
doc/rust.tex: rust.md doc/version.md
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js $< | \
	"$(CFG_PANDOC)" \
         --standalone --toc \
         --number-sections \
	     --include-before-body=doc/version.md \
         --from=markdown --to=latex \
         --output=$@

DOCS += doc/rustpkg.html
doc/rustpkg.html: rustpkg.md doc/version_info.html doc/rust.css doc/manual.css
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
	"$(CFG_PANDOC)" \
         --standalone --toc \
         --section-divs \
         --number-sections \
         --from=markdown --to=html \
         --css=rust.css \
         --css=manual.css \
	     --include-before-body=doc/version_info.html \
         --output=$@

DOCS += doc/tutorial.html
doc/tutorial.html: tutorial.md doc/version_info.html doc/rust.css
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
          $(CFG_PANDOC) --standalone --toc \
           --section-divs --number-sections \
           --from=markdown --to=html --css=rust.css \
	   --include-before-body=doc/version_info.html \
           --output=$@

DOCS_L10N += doc/l10n/ja/tutorial.html
doc/l10n/ja/tutorial.html: doc/l10n/ja/tutorial.md doc/version_info.html doc/rust.css
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight doc/l10n/ja/tutorial.md | \
          $(CFG_PANDOC) --standalone --toc \
           --section-divs --number-sections \
           --from=markdown --to=html --css=../../rust.css \
	   --include-before-body=doc/version_info.html \
           --output=$@

DOCS += doc/tutorial-macros.html
doc/tutorial-macros.html: tutorial-macros.md doc/version_info.html \
						  doc/rust.css
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
          $(CFG_PANDOC) --standalone --toc \
           --section-divs --number-sections \
           --from=markdown --to=html --css=rust.css \
	   --include-before-body=doc/version_info.html \
           --output=$@

DOCS += doc/tutorial-container.html
doc/tutorial-container.html: tutorial-container.md doc/version_info.html doc/rust.css
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
          $(CFG_PANDOC) --standalone --toc \
           --section-divs --number-sections \
           --from=markdown --to=html --css=rust.css \
	   --include-before-body=doc/version_info.html \
           --output=$@

DOCS += doc/tutorial-ffi.html
doc/tutorial-ffi.html: tutorial-ffi.md doc/version_info.html doc/rust.css
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
          $(CFG_PANDOC) --standalone --toc \
           --section-divs --number-sections \
           --from=markdown --to=html --css=rust.css \
	   --include-before-body=doc/version_info.html \
           --output=$@

DOCS += doc/tutorial-borrowed-ptr.html
doc/tutorial-borrowed-ptr.html: tutorial-borrowed-ptr.md doc/version_info.html doc/rust.css
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
          $(CFG_PANDOC) --standalone --toc \
           --section-divs --number-sections \
           --from=markdown --to=html --css=rust.css \
	   --include-before-body=doc/version_info.html \
           --output=$@

DOCS += doc/tutorial-tasks.html
doc/tutorial-tasks.html: tutorial-tasks.md doc/version_info.html doc/rust.css
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
          $(CFG_PANDOC) --standalone --toc \
           --section-divs --number-sections \
           --from=markdown --to=html --css=rust.css \
	   --include-before-body=doc/version_info.html \
           --output=$@

DOCS += doc/tutorial-conditions.html
doc/tutorial-conditions.html: tutorial-conditions.md doc/version_info.html doc/rust.css
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
          $(CFG_PANDOC) --standalone --toc \
           --section-divs --number-sections \
           --from=markdown --to=html --css=rust.css \
           --include-before-body=doc/version_info.html \
           --output=$@

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
doc/rust.pdf: doc/rust.tex
	@$(call E, pdflatex: $@)
	$(Q)$(CFG_PDFLATEX) \
        -interaction=batchmode \
        -output-directory=doc \
        $<

      endif
    endif
  endif

endif # No pandoc / node

######################################################################
# LLnextgen (grammar analysis from refman)
######################################################################
ifeq ($(CFG_LLNEXTGEN),)
  $(info cfg: no llnextgen found, omitting grammar-verification)
else
.PHONY: verify-grammar

doc/rust.g: rust.md $(S)src/etc/extract_grammar.py
	@$(call E, extract_grammar: $@)
	$(Q)$(CFG_PYTHON) $(S)src/etc/extract_grammar.py $< >$@

verify-grammar: doc/rust.g
	@$(call E, LLnextgen: $<)
	$(Q)$(CFG_LLNEXTGEN) --generate-lexer-wrapper=no $< >$@
	$(Q)rm -f doc/rust.c doc/rust.h
endif


######################################################################
# Rustdoc (libstd/extra)
######################################################################

ifeq ($(CFG_PANDOC),)
  $(info cfg: no pandoc found, omitting library doc build)
else

# The rustdoc executable
RUSTDOC = $(HBIN2_H_$(CFG_BUILD_TRIPLE))/rustdoc$(X_$(CFG_BUILD_TRIPLE))

# The library documenting macro
# $(1) - The output directory
# $(2) - The crate file
# $(3) - The crate soruce files
define libdoc
doc/$(1)/index.html: $(2) $(3) $$(RUSTDOC) doc/$(1)/rust.css
	@$$(call E, rustdoc: $$@)
	$(Q)$(RUSTDOC) $(2) --output-dir=doc/$(1)

doc/$(1)/rust.css: rust.css
	@$$(call E, cp: $$@)
	$(Q)cp $$< $$@

DOCS += doc/$(1)/index.html
endef

$(eval $(call libdoc,std,$(STDLIB_CRATE),$(STDLIB_INPUTS)))
$(eval $(call libdoc,extra,$(EXTRALIB_CRATE),$(EXTRALIB_INPUTS)))
endif


ifdef CFG_DISABLE_DOCS
  $(info cfg: disabling doc build (CFG_DISABLE_DOCS))
  DOCS :=
endif


doc/version.md: $(MKFILE_DEPS) $(wildcard $(S)doc/*.*)
	@$(call E, version-stamp: $@)
	$(Q)echo "$(CFG_VERSION)" >$@

doc/version_info.html: version_info.html.template $(MKFILE_DEPS) \
                       $(wildcard $(S)doc/*.*)
	@$(call E, version-info: $@)
	sed -e "s/VERSION/$(CFG_RELEASE)/; s/SHORT_HASH/$(shell echo \
                    $(CFG_VER_HASH) | head -c 8)/;\
	            s/STAMP/$(CFG_VER_HASH)/;" $< >$@

GENERATED += doc/version.md doc/version_info.html

docs: $(DOCS)

docs-l10n: $(DOCS_L10N)

doc/l10n/%.md: doc/po/%.md.po doc/po4a.conf
	po4a --copyright-holder="The Rust Project Developers" \
	     --package-name="Rust" \
	     --package-version="$(CFG_RELEASE)" \
	     -M UTF-8 -L UTF-8 \
	     doc/po4a.conf

.PHONY: docs-l10n
