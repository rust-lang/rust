-include ../tools.mk

all: $(call DYLIB,return1) $(call DYLIB,return2) $(call NATIVE_STATICLIB,return3)
	ls $(TMPDIR)
	$(RUSTC) --print cfg --target x86_64-unknown-linux-musl | grep crt-static

	$(RUSTC) no-deps.rs --cfg foo
	$(call RUN,no-deps)
	$(RUSTC) no-deps.rs --cfg bar
	$(call RUN,no-deps)

	$(RUSTC) dep.rs
	$(RUSTC) with-deps.rs --cfg foo
	$(call RUN,with-deps)
	$(RUSTC) with-deps.rs --cfg bar
	$(call RUN,with-deps)

	$(RUSTC) dep-with-staticlib.rs
	$(RUSTC) with-staticlib-deps.rs --cfg foo
	$(call RUN,with-staticlib-deps)
	$(RUSTC) with-staticlib-deps.rs --cfg bar
	$(call RUN,with-staticlib-deps)
