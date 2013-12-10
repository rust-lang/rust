# Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
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

BASE_DOC_OPTS := --from=markdown --standalone --toc --number-sections --include-before-body=doc/version_info.html
HTML_OPTS = $(BASE_DOC_OPTS) --to=html5  --section-divs --css=rust.css --include-in-header=doc/favicon.inc
TEX_OPTS = $(BASE_DOC_OPTS) --to=latex
EPUB_OPTS = $(BASE_DOC_OPTS) --to=epub

######################################################################
# Docs, from pandoc, rustdoc (which runs pandoc), and node
######################################################################

doc/rust.css: rust.css
	@$(call E, cp: $@)
	$(Q)cp -a $< $@ 2> /dev/null

doc/manual.inc: manual.inc
	@$(call E, cp: $@)
	$(Q)cp -a $< $@ 2> /dev/null

doc/favicon.inc: favicon.inc
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
doc/rust.html: rust.md doc/version_info.html doc/rust.css doc/manual.inc \
				doc/favicon.inc
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
	$(CFG_PANDOC) $(HTML_OPTS) --include-in-header=doc/manual.inc --output=$@

DOCS += doc/rust.tex
doc/rust.tex: rust.md doc/version.md
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js $< | \
	$(CFG_PANDOC) $(TEX_OPTS) --output=$@

DOCS += doc/rust.epub
doc/rust.epub: rust.md doc/version_info.html doc/rust.css
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
	$(CFG_PANDOC) $(EPUB_OPTS) --output=$@

DOCS += doc/rustpkg.html
doc/rustpkg.html: rustpkg.md doc/version_info.html doc/rust.css \
				doc/favicon.inc
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
	$(CFG_PANDOC) $(HTML_OPTS) --output=$@

DOCS += doc/rustdoc.html
doc/rustdoc.html: rustdoc.md doc/version_info.html doc/rust.css \
				doc/favicon.inc
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
	$(CFG_PANDOC) $(HTML_OPTS) --output=$@

DOCS += doc/tutorial.html
doc/tutorial.html: tutorial.md doc/version_info.html doc/rust.css \
				doc/favicon.inc
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
	$(CFG_PANDOC) $(HTML_OPTS) --output=$@

DOCS += doc/tutorial.tex
doc/tutorial.tex: tutorial.md doc/version.md
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js $< | \
	$(CFG_PANDOC) $(TEX_OPTS) --output=$@

DOCS += doc/tutorial.epub
doc/tutorial.epub: tutorial.md doc/version_info.html doc/rust.css
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
	$(CFG_PANDOC) $(EPUB_OPTS) --output=$@


DOCS_L10N += doc/l10n/ja/tutorial.html
doc/l10n/ja/tutorial.html: doc/l10n/ja/tutorial.md doc/version_info.html doc/rust.css
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight doc/l10n/ja/tutorial.md | \
          $(CFG_PANDOC) --standalone --toc \
           --section-divs --number-sections \
           --from=markdown --to=html5 --css=../../rust.css \
           --include-before-body=doc/version_info.html \
           --output=$@

DOCS += doc/tutorial-macros.html
doc/tutorial-macros.html: tutorial-macros.md doc/version_info.html doc/rust.css \
				doc/favicon.inc
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
	$(CFG_PANDOC) $(HTML_OPTS) --output=$@

DOCS += doc/tutorial-container.html
doc/tutorial-container.html: tutorial-container.md doc/version_info.html doc/rust.css \
				doc/favicon.inc
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
	$(CFG_PANDOC) $(HTML_OPTS) --output=$@

DOCS += doc/tutorial-ffi.html
doc/tutorial-ffi.html: tutorial-ffi.md doc/version_info.html doc/rust.css \
				doc/favicon.inc
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
	$(CFG_PANDOC) $(HTML_OPTS) --output=$@

DOCS += doc/tutorial-borrowed-ptr.html
doc/tutorial-borrowed-ptr.html: tutorial-borrowed-ptr.md doc/version_info.html doc/rust.css \
				doc/favicon.inc
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
	$(CFG_PANDOC) $(HTML_OPTS) --output=$@

DOCS += doc/tutorial-tasks.html
doc/tutorial-tasks.html: tutorial-tasks.md doc/version_info.html doc/rust.css \
				doc/favicon.inc
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
	$(CFG_PANDOC) $(HTML_OPTS) --output=$@

DOCS += doc/tutorial-conditions.html
doc/tutorial-conditions.html: tutorial-conditions.md doc/version_info.html doc/rust.css \
				doc/favicon.inc
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
	$(CFG_PANDOC) $(HTML_OPTS) --output=$@

DOCS += doc/tutorial-rustpkg.html
doc/tutorial-rustpkg.html: tutorial-rustpkg.md doc/version_info.html doc/rust.css \
				doc/favicon.inc
	@$(call E, pandoc: $@)
	$(Q)$(CFG_NODE) $(S)doc/prep.js --highlight $< | \
	$(CFG_PANDOC) $(HTML_OPTS) --output=$@

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

DOCS += doc/tutorial.pdf
doc/tutorial.pdf: doc/tutorial.tex
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

# The rustdoc executable
RUSTDOC = $(HBIN2_H_$(CFG_BUILD))/rustdoc$(X_$(CFG_BUILD))

# The library documenting macro
# $(1) - The crate name (std/extra)
# $(2) - The crate file
# $(3) - The relevant host build triple (to depend on libstd)
#
# Passes --cfg stage2 to rustdoc because it uses the stage2 librustc.
define libdoc
doc/$(1)/index.html: $$(RUSTDOC) $$(TLIB2_T_$(3)_H_$(3))/$(CFG_STDLIB_$(3))
	@$$(call E, rustdoc: $$@)
	$(Q)$(RUSTDOC) --cfg stage2 $(2)

DOCS += doc/$(1)/index.html
endef

$(eval $(call libdoc,std,$(STDLIB_CRATE),$(CFG_BUILD)))
$(eval $(call libdoc,extra,$(EXTRALIB_CRATE),$(CFG_BUILD)))


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
