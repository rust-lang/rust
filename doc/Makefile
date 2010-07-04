DOC_VER  := $(shell date +"%Y-%m-%d")-snap

all: rust.pdf rust.html

version.texi: Makefile
	git log -1 \
      --format='@macro gitversion%n%h %ci%n@end macro%n' >$@

%.pdf: %.texi version.texi
	texi2pdf $<

%.html: %.texi version.texi
	makeinfo --html --ifhtml --force --no-split --output=$@ $<

snap: rust.pdf
	mv $< rust-$(DOC_VER).pdf

clean:
	rm -f rust.aux rust.cp rust.fn rust.ky rust.log rust.pdf \
          rust.html rust.pg rust.toc rust.tp rust.vr rust.cps \
          version.texi
