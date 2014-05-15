-include ../tools.mk

FILES=f00.rs f01.rs f02.rs f03.rs f04.rs f05.rs f06.rs f07.rs \
      f08.rs f09.rs f10.rs f11.rs f12.rs f13.rs f14.rs f15.rs \
      f16.rs f17.rs f18.rs f19.rs f20.rs f21.rs f22.rs


# all: $(patsubst %.rs,$(TMPDIR)/%.dot,$(FILES)) $(patsubst %.rs,$(TMPDIR)/%.pp,$(FILES))
all: $(patsubst %.rs,$(TMPDIR)/%.check,$(FILES))


RUSTC_LIB=$(RUSTC) --crate-type=lib

define FIND_LAST_BLOCK
LASTBLOCKNUM_$(1) := $(shell $(RUSTC_LIB) --pretty=expanded,identified $(1) \
			 | grep block
			 | tail -1
			 | sed -e 's@.*/\* block \([0-9]*\) \*/.*@\1@')
endef

ifeq ($(findstring rustc,$(RUSTC)),)
$(error Must set RUSTC)
endif

$(TMPDIR)/%.pp: %.rs
	$(RUSTC_LIB) --pretty=expanded,identified $< -o $@

$(TMPDIR)/%.dot: %.rs
	$(eval $(call FIND_LAST_BLOCK,$<))
	$(RUSTC_LIB) --pretty flowgraph=$(LASTBLOCKNUM_$<) $< -o $@.tmp
	cat $@.tmp | sed -e 's@ (id=[0-9]*)@@g' \
                         -e 's@\[label=""\]@@' \
                         -e 's@digraph [a-zA-Z0-9_]* @digraph block @' \
                     > $@

$(TMPDIR)/%.check: %.rs $(TMPDIR)/%.dot
	diff -u $(patsubst %.rs,$(TMPDIR)/%.dot,$<) $(patsubst %.rs,%.dot-expected.dot,$<)
