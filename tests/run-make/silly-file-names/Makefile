# ignore-cross-compile we need to execute the binary
# ignore-windows we create files with < and > in their names

include ../tools.mk

all:
	echo '"comes from a file with a name that begins with <"' > "$(TMPDIR)/<leading-lt"
	echo '"comes from a file with a name that ends with >"' > "$(TMPDIR)/trailing-gt>"
	cp silly-file-names.rs "$(TMPDIR)/silly-file-names.rs"
	$(RUSTC) "$(TMPDIR)/silly-file-names.rs" -o "$(TMPDIR)/silly-file-names"
	"$(TMPDIR)/silly-file-names" > "$(TMPDIR)/silly-file-names.run.stdout"
	$(RUSTC_TEST_OP) "$(TMPDIR)/silly-file-names.run.stdout" silly-file-names.run.stdout
