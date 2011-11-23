######################################################################
# Doc variables and rules
######################################################################

docs: $(DOCS)

doc/keywords.texi: $(S)doc/keywords.txt $(S)src/etc/gen-keywords-table.py
	@$(call E, gen-keywords-table: $@)
	$(Q)$(S)src/etc/gen-keywords-table.py

doc/version.texi: $(MKFILES) rust.texi
	@$(call E, version-stamp: $@)
	$(Q)echo "@macro gitversion" >$@
	$(Q)echo "$(CFG_VERSION)" >>$@
	$(Q)echo "@end macro" >>$@

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

doc/std/index.html: nd/std/Languages.txt nd/std/Topics.txt nd/std/std.css \
                    $(STDLIB_CRATE) $(STDLIB_INPUTS)
	@$(call E, naturaldocs: $@)
	naturaldocs -i $(S)src/lib -o HTML doc/std -p nd/std -r -s Default std

nd/std/Languages.txt: $(S)doc/Languages.txt
	@$(call E, cp: $@)
	$(Q)cp $< $@

nd/std/Topics.txt: $(S)doc/Topics.txt
	@$(call E, cp: $@)
	$(Q)cp $< $@

nd/std/std.css: $(S)doc/std.css
	@$(call E, cp: $@)
	$(Q)cp $< $@
