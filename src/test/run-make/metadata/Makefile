-include ../tools.mk

# Linking the object and the metadata externally should produce a usable library
all: $(call RUN_BINFILE,baz)

$(TMPDIR)/foo.o: foo.rs
	$(RUSTC) -C prefer-dynamic --emit=obj $<

$(TMPDIR)/foo.metadata.o: foo.rs
	$(RUSTC) -C prefer-dynamic --emit=metadata $<

	# Make sure the output only contains metadata.
	nm "$(TMPDIR)/foo.metadata.o" | grep -vq bar
	nm "$(TMPDIR)/foo.metadata.o" | grep -q metadata

ifeq ($(UNAME),Darwin)
$(call DYLIB,foo): $(TMPDIR)/foo.o $(TMPDIR)/foo.metadata.o
	$(CC) -dynamiclib -Wl,dylib -o $@ $^
else ifdef IS_MSVC
$(call DYLIB,foo): $(TMPDIR)/foo.o $(TMPDIR)/foo.metadata.o
	$(CC) $^ -link -dll -out:`cygpath -w $@`
else
$(call DYLIB,foo): $(TMPDIR)/foo.o $(TMPDIR)/foo.metadata.o
	$(CC) -shared -o $@ $^
endif

$(call RUN_BINFILE,baz): baz.rs $(call DYLIB,foo)
	$(RUSTC) $<
