######################################################################
# Doc variables and rules
######################################################################

doc/version.texi: $(MKFILES) rust.texi
	echo "@macro gitversion" >$@
	echo "$(CFG_VERSION)" >>$@
	echo "@end macro" >>$@

doc/%.pdf: %.texi doc/version.texi
	texi2pdf --batch -I doc -o $@ --clean $<

doc/%.html: %.texi doc/version.texi
	makeinfo -I doc --html --ifhtml --force --no-split --output=$@ $<

docsnap: doc/rust.pdf
	mv $< doc/rust-$(shell date +"%Y-%m-%d")-snap.pdf
