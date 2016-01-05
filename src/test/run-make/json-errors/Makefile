-include ../tools.mk

all:
	cp foo.rs $(TMPDIR)
	cd $(TMPDIR)
	$(RUSTC) -Z unstable-options --output=json foo.rs 2>foo.log || true
	grep -q '{"message":"unresolved name `y`","code":{"code":"E0425","explanation":"\\nAn unresolved name was used. Example of erroneous codes.*"},"level":"error","span":{"file_name":"foo.rs","byte_start":523,"byte_end":524,"line_start":14,"line_end":14,"column_start":18,"column_end":19},"children":\[\]}' foo.log
