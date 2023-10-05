include ../tools.mk

# We are creating files with forbidden characters in their names, so we need to
# ignore-windows

ifdef RUSTC_BLESS_TEST
    RUSTC_TEST_OP = cp
else
    RUSTC_TEST_OP = $(DIFF)
endif

all:
	echo '"comes from a file with a name that begins with <"' > "$(TMPDIR)/<leading-lt"
	echo '"comes from a file with a name that ends with >"' > "$(TMPDIR)/trailing-gt>"
	cp silly-file-names.rs "$(TMPDIR)/silly-file-names.rs"
	$(RUSTC) "$(TMPDIR)/silly-file-names.rs" -o "$(TMPDIR)/silly-file-names"
	"$(TMPDIR)/silly-file-names" > "$(TMPDIR)/silly-file-names.run.stdout"
	$(RUSTC_TEST_OP) "$(TMPDIR)/silly-file-names.run.stdout" silly-file-names.run.stdout
