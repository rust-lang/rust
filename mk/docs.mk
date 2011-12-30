######################################################################
# Doc variables and rules
######################################################################

doc/keywords.texi: $(S)doc/keywords.txt $(S)src/etc/gen-keywords-table.py
	@$(call E, gen-keywords-table: $@)
	$(Q)$(S)src/etc/gen-keywords-table.py

doc/version.texi: $(MKFILE_DEPS) rust.texi
	@$(call E, version-stamp: $@)
	$(Q)echo "@macro gitversion" >$@
	$(Q)echo "$(CFG_VERSION)" >>$@
	$(Q)echo "@end macro" >>$@

GENERATED += doc/keywords.texi doc/version.texi

doc/%.pdf: %.texi doc/version.texi doc/keywords.texi
	@$(call E, texi2pdf: $@)
	@# LC_COLLATE=C works around a bug in texi2dvi; see
	@# https://bugzilla.redhat.com/show_bug.cgi?id=583011 and
	@# https://github.com/graydon/rust/issues/1134
	$(Q)LC_COLLATE=C texi2pdf --silent --batch -I doc -o $@ --clean $<

doc/%.html: %.texi doc/version.texi doc/keywords.texi
	@$(call E, makeinfo: $@)
	$(Q)makeinfo -I doc --html --ifhtml --force --no-split --output=$@ $<

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

DOCS += nd/$(1)/index.html nd/$(1)/lib.css

endef

$(eval $(call libdoc,core,$(CORELIB_CRATE) $(CORELIB_INPUTS)))
$(eval $(call libdoc,std,$(STDLIB_CRATE) $(STDLIB_INPUTS)))

docs: $(DOCS)
