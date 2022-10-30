include ../tools.mk

# ignore-musl
# Objects are reproducible but their path is not.

all:  \
	smoke \
	debug \
	opt \
	link_paths \
	remap_paths \
	different_source_dirs_rlib \
	remap_cwd_rlib \
	remap_cwd_to_empty \
	extern_flags

# TODO: Builds of `bin` crate types are not deterministic with debuginfo=2 on
# Windows.
# See: https://github.com/rust-lang/rust/pull/87320#issuecomment-920105533
# Issue: https://github.com/rust-lang/rust/issues/88982
#
#	different_source_dirs_bin \
#	remap_cwd_bin \

smoke:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) linker.rs -O
	$(RUSTC) reproducible-build-aux.rs
	$(RUSTC) reproducible-build.rs -C linker=$(call RUN_BINFILE,linker)
	$(RUSTC) reproducible-build.rs -C linker=$(call RUN_BINFILE,linker)
	diff -u "$(TMPDIR)/linker-arguments1" "$(TMPDIR)/linker-arguments2"

debug:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) linker.rs -O
	$(RUSTC) reproducible-build-aux.rs -g
	$(RUSTC) reproducible-build.rs -C linker=$(call RUN_BINFILE,linker) -g
	$(RUSTC) reproducible-build.rs -C linker=$(call RUN_BINFILE,linker) -g
	diff -u "$(TMPDIR)/linker-arguments1" "$(TMPDIR)/linker-arguments2"

opt:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) linker.rs -O
	$(RUSTC) reproducible-build-aux.rs -O
	$(RUSTC) reproducible-build.rs -C linker=$(call RUN_BINFILE,linker) -O
	$(RUSTC) reproducible-build.rs -C linker=$(call RUN_BINFILE,linker) -O
	diff -u "$(TMPDIR)/linker-arguments1" "$(TMPDIR)/linker-arguments2"

link_paths:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) reproducible-build-aux.rs
	$(RUSTC) reproducible-build.rs --crate-type rlib -L /b
	cp $(TMPDIR)/libreproducible_build.rlib $(TMPDIR)/libfoo.rlib
	$(RUSTC) reproducible-build.rs --crate-type rlib -L /a
	cmp "$(TMPDIR)/libreproducible_build.rlib" "$(TMPDIR)/libfoo.rlib" || exit 1

remap_paths:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) reproducible-build-aux.rs
	$(RUSTC) reproducible-build.rs --crate-type rlib --remap-path-prefix=/a=/c
	cp $(TMPDIR)/libreproducible_build.rlib $(TMPDIR)/libfoo.rlib
	$(RUSTC) reproducible-build.rs --crate-type rlib --remap-path-prefix=/b=/c
	cmp "$(TMPDIR)/libreproducible_build.rlib" "$(TMPDIR)/libfoo.rlib" || exit 1

different_source_dirs_bin:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) reproducible-build-aux.rs
	mkdir $(TMPDIR)/test
	cp reproducible-build.rs $(TMPDIR)/test
	$(RUSTC) reproducible-build.rs --crate-type bin --remap-path-prefix=$$PWD=/b
	cp $(TMPDIR)/reproducible-build $(TMPDIR)/foo
	(cd $(TMPDIR)/test && $(RUSTC) reproducible-build.rs \
		--remap-path-prefix=$(TMPDIR)/test=/b \
		--crate-type bin)
	cmp "$(TMPDIR)/reproducible-build" "$(TMPDIR)/foo" || exit 1

different_source_dirs_rlib:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) reproducible-build-aux.rs
	mkdir $(TMPDIR)/test
	cp reproducible-build.rs $(TMPDIR)/test
	$(RUSTC) reproducible-build.rs --crate-type rlib --remap-path-prefix=$$PWD=/b
	cp $(TMPDIR)/libreproducible_build.rlib $(TMPDIR)/libfoo.rlib
	(cd $(TMPDIR)/test && $(RUSTC) reproducible-build.rs \
		--remap-path-prefix=$(TMPDIR)/test=/b \
		--crate-type rlib)
	cmp "$(TMPDIR)/libreproducible_build.rlib" "$(TMPDIR)/libfoo.rlib" || exit 1

remap_cwd_bin:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) reproducible-build-aux.rs
	mkdir $(TMPDIR)/test
	cp reproducible-build.rs $(TMPDIR)/test
	$(RUSTC) reproducible-build.rs --crate-type bin -C debuginfo=2 \
	  -Z remap-cwd-prefix=.
	cp $(TMPDIR)/reproducible-build $(TMPDIR)/first
	(cd $(TMPDIR)/test && \
	 $(RUSTC) reproducible-build.rs --crate-type bin -C debuginfo=2 \
	   -Z remap-cwd-prefix=.)
	cmp "$(TMPDIR)/first" "$(TMPDIR)/reproducible-build" || exit 1

remap_cwd_rlib:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) reproducible-build-aux.rs
	mkdir $(TMPDIR)/test
	cp reproducible-build.rs $(TMPDIR)/test
	$(RUSTC) reproducible-build.rs --crate-type rlib -C debuginfo=2 \
	  -Z remap-cwd-prefix=.
	cp $(TMPDIR)/libreproducible_build.rlib $(TMPDIR)/libfirst.rlib
	(cd $(TMPDIR)/test && \
	 $(RUSTC) reproducible-build.rs --crate-type rlib -C debuginfo=2 \
	   -Z remap-cwd-prefix=.)
	cmp "$(TMPDIR)/libfirst.rlib" "$(TMPDIR)/libreproducible_build.rlib" || exit 1

remap_cwd_to_empty:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) reproducible-build-aux.rs
	mkdir $(TMPDIR)/test
	cp reproducible-build.rs $(TMPDIR)/test
	$(RUSTC) reproducible-build.rs --crate-type rlib -C debuginfo=2 \
	  -Z remap-cwd-prefix=
	cp $(TMPDIR)/libreproducible_build.rlib $(TMPDIR)/libfirst.rlib
	(cd $(TMPDIR)/test && \
	 $(RUSTC) reproducible-build.rs --crate-type rlib -C debuginfo=2 \
	   -Z remap-cwd-prefix=)
	cmp "$(TMPDIR)/libfirst.rlib" "$(TMPDIR)/libreproducible_build.rlib" || exit 1

extern_flags:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) reproducible-build-aux.rs
	$(RUSTC) reproducible-build.rs \
		--extern reproducible_build_aux=$(TMPDIR)/libreproducible_build_aux.rlib \
		--crate-type rlib
	cp $(TMPDIR)/libreproducible_build_aux.rlib $(TMPDIR)/libbar.rlib
	cp $(TMPDIR)/libreproducible_build.rlib $(TMPDIR)/libfoo.rlib
	$(RUSTC) reproducible-build.rs \
		--extern reproducible_build_aux=$(TMPDIR)/libbar.rlib \
		--crate-type rlib
	cmp "$(TMPDIR)/libreproducible_build.rlib" "$(TMPDIR)/libfoo.rlib" || exit 1
