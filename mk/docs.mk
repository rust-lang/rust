######################################################################
# Doc variables and rules
######################################################################

doc/version.texi: $(MKFILES) rust.texi
	@$(call E, version-stamp: $@)
	$(Q)echo "@macro gitversion" >$@
	$(Q)echo "$(CFG_VERSION)" >>$@
	$(Q)echo "@end macro" >>$@

doc/%.pdf: %.texi doc/version.texi
	@$(call E, texi2pdf: $@)
	$(Q)texi2pdf --silent --batch -I doc -o $@ --clean $<

doc/%.html: %.texi doc/version.texi
	@$(call E, makeinfo: $@)
	$(Q)makeinfo -I doc --html --ifhtml --force --no-split --output=$@ $<

docsnap: doc/rust.pdf
	@$(call E, snap: doc/rust-$(shell date +"%Y-%m-%d")-snap.pdf)
	$(Q)mv $< doc/rust-$(shell date +"%Y-%m-%d")-snap.pdf
