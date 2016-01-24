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
# The various pieces of standalone documentation.
#
# The DOCS variable is their names (with no file extension).
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
DOCS := index \
    complement-lang-faq complement-design-faq complement-project-faq \
    rustdoc reference grammar

# Legacy guides, preserved for a while to reduce the number of 404s
DOCS += guide-crates guide-error-handling guide-ffi guide-macros guide \
    guide-ownership guide-plugins guide-pointers guide-strings guide-tasks \
    guide-testing tutorial intro


RUSTDOC_DEPS_reference := doc/full-toc.inc
RUSTDOC_FLAGS_reference := --html-in-header=doc/full-toc.inc

L10N_LANGS := ja

# Generally no need to edit below here.

# The options are passed to the documentation generators.
RUSTDOC_HTML_OPTS_NO_CSS = --html-before-content=doc/version_info.html \
	--html-in-header=doc/favicon.inc \
	--html-after-content=doc/footer.inc \
	--markdown-playground-url='https://play.rust-lang.org/'

RUSTDOC_HTML_OPTS = $(RUSTDOC_HTML_OPTS_NO_CSS) --markdown-css rust.css

# The rustdoc executable...
RUSTDOC_EXE = $(HBIN2_H_$(CFG_BUILD))/rustdoc$(X_$(CFG_BUILD))
# ...with rpath included in case --disable-rpath was provided to
# ./configure
RUSTDOC = $(RPATH_VAR2_T_$(CFG_BUILD)_H_$(CFG_BUILD)) $(RUSTDOC_EXE)

# The rustbook executable...
RUSTBOOK_EXE = $(HBIN2_H_$(CFG_BUILD))/rustbook$(X_$(CFG_BUILD))
# ...with rpath included in case --disable-rpath was provided to
# ./configure
RUSTBOOK = $(RPATH_VAR2_T_$(CFG_BUILD)_H_$(CFG_BUILD)) $(RUSTBOOK_EXE)

# The error-index-generator executable...
ERR_IDX_GEN_EXE = $(HBIN2_H_$(CFG_BUILD))/error-index-generator$(X_$(CFG_BUILD))
ERR_IDX_GEN = $(RPATH_VAR2_T_$(CFG_BUILD)_H_$(CFG_BUILD)) $(ERR_IDX_GEN_EXE)

D := $(S)src/doc

DOC_TARGETS := book nomicon style error-index
COMPILER_DOC_TARGETS :=
DOC_L10N_TARGETS :=

# If NO_REBUILD is set then break the dependencies on rustdoc so we
# build the documentation without having to rebuild rustdoc.
ifeq ($(NO_REBUILD),)
HTML_DEPS := $(RUSTDOC_EXE)
else
HTML_DEPS :=
endif

######################################################################
# Rust version
######################################################################

HTML_DEPS += doc/version_info.html
doc/version_info.html: $(D)/version_info.html.template $(MKFILE_DEPS) \
                       $(wildcard $(D)/*.*) | doc/
	@$(call E, version-info: $@)
	$(Q)sed -e "s/VERSION/$(CFG_RELEASE)/; \
                s/SHORT_HASH/$(CFG_SHORT_VER_HASH)/; \
                s/STAMP/$(CFG_VER_HASH)/;" $< >$@

GENERATED += doc/version_info.html

######################################################################
# Docs from rustdoc
######################################################################

doc/:
	@mkdir -p $@

HTML_DEPS += doc/rust.css
doc/rust.css: $(D)/rust.css | doc/
	@$(call E, cp: $@)
	$(Q)cp -PRp $< $@ 2> /dev/null

HTML_DEPS += doc/favicon.inc
doc/favicon.inc: $(D)/favicon.inc | doc/
	@$(call E, cp: $@)
	$(Q)cp -PRp $< $@ 2> /dev/null

doc/full-toc.inc: $(D)/full-toc.inc | doc/
	@$(call E, cp: $@)
	$(Q)cp -PRp $< $@ 2> /dev/null

HTML_DEPS += doc/footer.inc
doc/footer.inc: $(D)/footer.inc | doc/
	@$(call E, cp: $@)
	$(Q)cp -PRp $< $@ 2> /dev/null

# The (english) documentation for each doc item.
DOC_TARGETS += doc/not_found.html
doc/not_found.html: $(D)/not_found.md $(HTML_DEPS) | doc/
	@$(call E, rustdoc: $@)
	$(Q)$(RUSTDOC) $(RUSTDOC_HTML_OPTS_NO_CSS) \
		--markdown-no-toc \
		--markdown-css https://doc.rust-lang.org/rust.css $<

define DEF_DOC

# HTML (rustdoc)
DOC_TARGETS += doc/$(1).html
doc/$(1).html: $$(D)/$(1).md $$(HTML_DEPS) $$(RUSTDOC_DEPS_$(1)) | doc/
	@$$(call E, rustdoc: $$@)
	$$(Q)$$(RUSTDOC) $$(RUSTDOC_HTML_OPTS) $$(RUSTDOC_FLAGS_$(1)) $$<

endef

$(foreach docname,$(DOCS),$(eval $(call DEF_DOC,$(docname))))


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
		$$(TLIB2_T_$(CFG_BUILD)_H_$(CFG_BUILD))/stamp.$$(dep)) \
	$$(foreach dep,$$(filter $$(DOC_CRATES), $$(RUST_DEPS_$(1))), \
		doc/$$(dep)/)
else
LIB_DOC_DEP_$(1) = $$(CRATEFILE_$(1)) $$(RSINPUTS_$(1))
endif

doc/$(1)/:
	$$(Q)mkdir -p $$@

doc/$(1)/index.html: CFG_COMPILER_HOST_TRIPLE = $(CFG_TARGET)
doc/$(1)/index.html: $$(LIB_DOC_DEP_$(1)) doc/$(1)/
	@$$(call E, rustdoc: $$@)
	$$(Q)CFG_LLVM_LINKAGE_FILE=$$(LLVM_LINKAGE_PATH_$(CFG_BUILD)) \
		$$(RUSTDOC) --cfg dox --cfg stage2 $$(RUSTFLAGS_$(1)) $$<
endef

$(foreach crate,$(CRATES),$(eval $(call DEF_LIB_DOC,$(crate))))

COMPILER_DOC_TARGETS := $(CRATES:%=doc/%/index.html)
ifdef CFG_ENABLE_COMPILER_DOCS
  DOC_TARGETS += $(COMPILER_DOC_TARGETS)
else
  DOC_TARGETS += $(DOC_CRATES:%=doc/%/index.html)
endif

ifdef CFG_DISABLE_DOCS
  $(info cfg: disabling doc build (CFG_DISABLE_DOCS))
  DOC_TARGETS :=
  COMPILER_DOC_TARGETS :=
endif

docs: $(DOC_TARGETS)
doc: docs
compiler-docs: $(COMPILER_DOC_TARGETS)

book: doc/book/index.html

doc/book/index.html: $(RUSTBOOK_EXE) $(wildcard $(S)/src/doc/book/*.md) | doc/
	@$(call E, rustbook: $@)
	$(Q)rm -rf doc/book
	$(Q)$(RUSTBOOK) build $(S)src/doc/book doc/book

nomicon: doc/nomicon/index.html

doc/nomicon/index.html: $(RUSTBOOK_EXE) $(wildcard $(S)/src/doc/nomicon/*.md) | doc/
	@$(call E, rustbook: $@)
	$(Q)rm -rf doc/nomicon
	$(Q)$(RUSTBOOK) build $(S)src/doc/nomicon doc/nomicon

style: doc/style/index.html

doc/style/index.html: $(RUSTBOOK_EXE) $(wildcard $(S)/src/doc/style/*.md) | doc/
	@$(call E, rustbook: $@)
	$(Q)rm -rf doc/style
	$(Q)$(RUSTBOOK) build $(S)src/doc/style doc/style

error-index: doc/error-index.html

doc/error-index.html: $(ERR_IDX_GEN_EXE) | doc/
	$(Q)$(call E, error-index-generator: $@)
	$(Q)$(ERR_IDX_GEN)
