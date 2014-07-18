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
# The various pieces of standalone documentation: guides, tutorial,
# manual etc.
#
# The DOCS variable is their names (with no file extension).
#
# PDF_DOCS lists the targets for which PDF documentation should be
# build.
#
# RUSTDOC_FLAGS_xyz variables are extra arguments to pass to the
# rustdoc invocation for xyz.
#
# RUSTDOC_DEPS_xyz are extra dependencies for the rustdoc invocation
# on xyz.
#
# L10N_LANGS are the languages for which the docs have been
# translated.
######################################################################
DOCS := index intro tutorial guide guide-ffi guide-macros guide-lifetimes \
	guide-tasks guide-container guide-pointers guide-testing \
	guide-runtime complement-bugreport \
	complement-lang-faq complement-design-faq complement-project-faq rust \
    rustdoc guide-unsafe guide-strings

PDF_DOCS := tutorial rust

RUSTDOC_DEPS_rust := doc/full-toc.inc
RUSTDOC_FLAGS_rust := --html-in-header=doc/full-toc.inc

L10N_LANGS := ja

# Generally no need to edit below here.

# The options are passed to the documentation generators.
RUSTDOC_HTML_OPTS_NO_CSS = --html-before-content=doc/version_info.html \
	--html-in-header=doc/favicon.inc \
	--html-after-content=doc/footer.inc \
	--markdown-playground-url='http://play.rust-lang.org/'

RUSTDOC_HTML_OPTS = $(RUSTDOC_HTML_OPTS_NO_CSS) --markdown-css rust.css

PANDOC_BASE_OPTS := --standalone --toc --number-sections
PANDOC_TEX_OPTS = $(PANDOC_BASE_OPTS) --include-before-body=doc/version.tex \
	--from=markdown --include-before-body=doc/footer.tex --to=latex
PANDOC_EPUB_OPTS = $(PANDOC_BASE_OPTS) --to=epub

# The rustdoc executable...
RUSTDOC_EXE = $(HBIN2_H_$(CFG_BUILD))/rustdoc$(X_$(CFG_BUILD))
# ...with rpath included in case --disable-rpath was provided to
# ./configure
RUSTDOC = $(RPATH_VAR2_T_$(CFG_BUILD)_H_$(CFG_BUILD)) $(RUSTDOC_EXE)

D := $(S)src/doc

DOC_TARGETS :=
COMPILER_DOC_TARGETS :=
DOC_L10N_TARGETS :=

# If NO_REBUILD is set then break the dependencies on rustdoc so we
# build the documentation without having to rebuild rustdoc.
ifeq ($(NO_REBUILD),)
HTML_DEPS := $(RUSTDOC_EXE)
else
HTML_DEPS :=
endif

# Check for the various external utilities for the EPUB/PDF docs:

ifeq ($(CFG_PDFLATEX),)
  $(info cfg: no pdflatex found, deferring to xelatex)
  ifeq ($(CFG_XELATEX),)
    $(info cfg: no xelatex found, deferring to lualatex)
    ifeq ($(CFG_LUALATEX),)
      $(info cfg: no lualatex found, disabling LaTeX docs)
      NO_PDF_DOCS = 1
	else
      CFG_LATEX := $(CFG_LUALATEX)
    endif
  else
    CFG_LATEX := $(CFG_XELATEX)
  endif
else
  CFG_LATEX := $(CFG_PDFLATEX)
endif


ifeq ($(CFG_PANDOC),)
$(info cfg: no pandoc found, omitting PDF and EPUB docs)
ONLY_HTML_DOCS = 1
endif


######################################################################
# Rust version
######################################################################

doc/version.tex: $(MKFILE_DEPS) $(wildcard $(D)/*.*) | doc/
	@$(call E, version-stamp: $@)
	$(Q)echo "$(CFG_VERSION)" >$@

HTML_DEPS += doc/version_info.html
doc/version_info.html: $(D)/version_info.html.template $(MKFILE_DEPS) \
                       $(wildcard $(D)/*.*) | doc/
	@$(call E, version-info: $@)
	$(Q)sed -e "s/VERSION/$(CFG_RELEASE)/; s/SHORT_HASH/$(shell echo \
                    $(CFG_VER_HASH) | head -c 8)/;\
                s/STAMP/$(CFG_VER_HASH)/;" $< >$@

GENERATED += doc/version.tex doc/version_info.html

######################################################################
# Docs, from rustdoc and sometimes pandoc
######################################################################

doc/:
	@mkdir -p $@

HTML_DEPS += doc/rust.css
doc/rust.css: $(D)/rust.css | doc/
	@$(call E, cp: $@)
	$(Q)cp -a $< $@ 2> /dev/null

HTML_DEPS += doc/favicon.inc
doc/favicon.inc: $(D)/favicon.inc | doc/
	@$(call E, cp: $@)
	$(Q)cp -a $< $@ 2> /dev/null

doc/full-toc.inc: $(D)/full-toc.inc | doc/
	@$(call E, cp: $@)
	$(Q)cp -a $< $@ 2> /dev/null

HTML_DEPS += doc/footer.inc
doc/footer.inc: $(D)/footer.inc | doc/
	@$(call E, cp: $@)
	$(Q)cp -a $< $@ 2> /dev/null

# The (english) documentation for each doc item.

define DEF_SHOULD_BUILD_PDF_DOC
SHOULD_BUILD_PDF_DOC_$(1) = 1
endef
$(foreach docname,$(PDF_DOCS),$(eval $(call DEF_SHOULD_BUILD_PDF_DOC,$(docname))))

doc/footer.tex: $(D)/footer.inc | doc/
	@$(call E, pandoc: $@)
	$(CFG_PANDOC) --from=html --to=latex $< --output=$@

# HTML (rustdoc)
DOC_TARGETS += doc/not_found.html
doc/not_found.html: $(D)/not_found.md $(HTML_DEPS) | doc/
	@$(call E, rustdoc: $@)
	$(Q)$(RUSTDOC) $(RUSTDOC_HTML_OPTS_NO_CSS) \
		--markdown-css http://doc.rust-lang.org/rust.css $<

define DEF_DOC

# HTML (rustdoc)
DOC_TARGETS += doc/$(1).html
doc/$(1).html: $$(D)/$(1).md $$(HTML_DEPS) $$(RUSTDOC_DEPS_$(1)) | doc/
	@$$(call E, rustdoc: $$@)
	$$(Q)$$(RUSTDOC) $$(RUSTDOC_HTML_OPTS) $$(RUSTDOC_FLAGS_$(1)) $$<

ifneq ($(ONLY_HTML_DOCS),1)

# EPUB (pandoc directly)
DOC_TARGETS += doc/$(1).epub
doc/$(1).epub: $$(D)/$(1).md | doc/
	@$$(call E, pandoc: $$@)
	$$(CFG_PANDOC) $$(PANDOC_EPUB_OPTS) $$< --output=$$@

# PDF (md =(pandoc)=> tex =(pdflatex)=> pdf)
DOC_TARGETS += doc/$(1).tex
doc/$(1).tex: $$(D)/$(1).md doc/footer.tex doc/version.tex | doc/
	@$$(call E, pandoc: $$@)
	$$(CFG_PANDOC) $$(PANDOC_TEX_OPTS) $$< --output=$$@

ifneq ($(NO_PDF_DOCS),1)
ifeq ($$(SHOULD_BUILD_PDF_DOC_$(1)),1)
DOC_TARGETS += doc/$(1).pdf
doc/$(1).pdf: doc/$(1).tex
	@$$(call E, latex compiler: $$@)
	$$(Q)$$(CFG_LATEX) \
	-interaction=batchmode \
	-output-directory=doc \
	$$<
endif # SHOULD_BUILD_PDF_DOCS_$(1)
endif # NO_PDF_DOCS

endif # ONLY_HTML_DOCS

endef

$(foreach docname,$(DOCS),$(eval $(call DEF_DOC,$(docname))))


# Localized documentation

# FIXME: I (huonw) haven't actually been able to test properly, since
# e.g. (by default) I'm doing an out-of-tree build (#12763), but even
# adjusting for that, the files are too old(?) and are rejected by
# po4a.
#
# As such, I've attempted to get it working as much as possible (and
# switching from pandoc to rustdoc), but preserving the old behaviour
# (e.g. only running on the tutorial)
.PHONY: l10n-mds
l10n-mds: $(D)/po4a.conf \
		$(foreach lang,$(L10N_LANG),$(D)/po/$(lang)/*.md.po)
	$(warning WARNING: localized documentation is experimental)
	po4a --copyright-holder="The Rust Project Developers" \
		--package-name="Rust" \
		--package-version="$(CFG_RELEASE)" \
		-M UTF-8 -L UTF-8 \
		$(D)/po4a.conf

define DEF_L10N_DOC
DOC_L10N_TARGETS += doc/l10n/$(1)/$(2).html
doc/l10n/$(1)/$(2).html: l10n-mds $$(HTML_DEPS) $$(RUSTDOC_DEPS_$(2))
	@$$(call E, rustdoc: $$@)
	$$(RUSTDOC) $$(RUSTDOC_HTML_OPTS) $$(RUSTDOC_FLAGS_$(1)) doc/l10n/$(1)/$(2).md
endef

$(foreach lang,$(L10N_LANGS),$(eval $(call DEF_L10N_DOC,$(lang),tutorial)))


######################################################################
# LLnextgen (grammar analysis from refman)
######################################################################

ifeq ($(CFG_LLNEXTGEN),)
  $(info cfg: no llnextgen found, omitting grammar-verification)
else
.PHONY: verify-grammar

doc/rust.g: $(D)/rust.md $(S)src/etc/extract_grammar.py
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


# The library documenting macro
#
# $(1) - The crate name (std/extra)
#
# Passes --cfg stage2 to rustdoc because it uses the stage2 librustc.
define DEF_LIB_DOC

# If NO_REBUILD is set then break the dependencies on rustdoc so we
# build crate documentation without having to rebuild rustdoc.
ifeq ($(NO_REBUILD),)
LIB_DOC_DEP_$(1) = \
	$$(CRATEFILE_$(1)) \
	$$(RSINPUTS_$(1)) \
	$$(RUSTDOC_EXE) \
	$$(foreach dep,$$(RUST_DEPS_$(1)), \
		$$(TLIB2_T_$(CFG_BUILD)_H_$(CFG_BUILD))/stamp.$$(dep) \
		doc/$$(dep)/)
else
LIB_DOC_DEP_$(1) = $$(CRATEFILE_$(1)) $$(RSINPUTS_$(1))
endif

doc/$(1)/:
	$$(Q)mkdir -p $$@

$(2) += doc/$(1)/index.html
doc/$(1)/index.html: CFG_COMPILER_HOST_TRIPLE = $(CFG_TARGET)
doc/$(1)/index.html: $$(LIB_DOC_DEP_$(1)) doc/$(1)/
	@$$(call E, rustdoc $$@)
	$$(Q)$$(RUSTDOC) --cfg dox --cfg stage2 $$<
endef

$(foreach crate,$(DOC_CRATES),$(eval $(call DEF_LIB_DOC,$(crate),DOC_TARGETS)))
$(foreach crate,$(COMPILER_DOC_CRATES),$(eval $(call DEF_LIB_DOC,$(crate),COMPILER_DOC_TARGETS)))

ifdef CFG_DISABLE_DOCS
  $(info cfg: disabling doc build (CFG_DISABLE_DOCS))
  DOC_TARGETS :=
  COMPILER_DOC_TARGETS :=
endif

docs: $(DOC_TARGETS)
compiler-docs: $(COMPILER_DOC_TARGETS)

docs-l10n: $(DOC_L10N_TARGETS)

.PHONY: docs-l10n
