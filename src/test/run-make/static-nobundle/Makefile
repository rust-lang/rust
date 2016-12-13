-include ../tools.mk

all: $(call NATIVE_STATICLIB,foo)
	$(RUSTC) bar.rs

	# Check that libbar.rlib does not contain the definition of `func`
	nm $(TMPDIR)/libbar.rlib | (! grep "T _*func")
	nm $(TMPDIR)/libbar.rlib | grep "U _*func"

	# Check that foo gets passed to the linker (as either `-l foo` or `foo.lib`)
	$(RUSTC) main.rs -Z print-link-args | grep -e "-l[\" ]*foo" -e "foo.lib"

	$(call RUN,main)
