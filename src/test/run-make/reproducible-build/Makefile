-include ../tools.mk
all:  \
	smoke \
	debug \
	opt \
	link_paths \
	remap_paths \
	different_source_dirs \
	extern_flags

smoke:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) reproducible-build-aux.rs
	$(RUSTC) reproducible-build.rs -o"$(TMPDIR)/reproducible-build1"
	$(RUSTC) reproducible-build.rs -o"$(TMPDIR)/reproducible-build2"
	$(B2)
	nm "$(TMPDIR)/reproducible-build1" | sort > "$(TMPDIR)/reproducible-build1.nm"
	nm "$(TMPDIR)/reproducible-build2" | sort > "$(TMPDIR)/reproducible-build2.nm"
	cmp "$(TMPDIR)/reproducible-build1.nm" "$(TMPDIR)/reproducible-build2.nm" || exit 1

debug:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) reproducible-build-aux.rs -g
	$(RUSTC) reproducible-build.rs -o"$(TMPDIR)/reproducible-build1" -g
	$(RUSTC) reproducible-build.rs -o"$(TMPDIR)/reproducible-build2" -g
	nm "$(TMPDIR)/reproducible-build1" | sort > "$(TMPDIR)/reproducible-build1-debug.nm"
	nm "$(TMPDIR)/reproducible-build2" | sort > "$(TMPDIR)/reproducible-build2-debug.nm"
	cmp "$(TMPDIR)/reproducible-build1-debug.nm" "$(TMPDIR)/reproducible-build2-debug.nm" || exit 1

opt:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) reproducible-build-aux.rs -O
	$(RUSTC) reproducible-build.rs -o"$(TMPDIR)/reproducible-build1" -O
	$(RUSTC) reproducible-build.rs -o"$(TMPDIR)/reproducible-build2" -O
	nm "$(TMPDIR)/reproducible-build1" | sort > "$(TMPDIR)/reproducible-build1-opt.nm"
	nm "$(TMPDIR)/reproducible-build2" | sort > "$(TMPDIR)/reproducible-build2-opt.nm"
	cmp "$(TMPDIR)/reproducible-build1-opt.nm" "$(TMPDIR)/reproducible-build2-opt.nm" || exit 1

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

different_source_dirs:
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
