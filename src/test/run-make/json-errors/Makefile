-include ../tools.mk

all:
	cp foo.rs $(TMPDIR)
	cd $(TMPDIR)
	-$(RUSTC) -Z unstable-options --error-format=json foo.rs 2>foo.log
	grep -q '{"message":"unresolved name `y`","code":{"code":"E0425","explanation":"\\nAn unresolved name was used. Example of erroneous codes.*"},"level":"error","span":{"file_name":"foo.rs","byte_start":496,"byte_end":497,"line_start":12,"line_end":12,"column_start":18,"column_end":19},"children":\[\]}' foo.log
	grep -q '{"message":".*","code":{"code":"E0277","explanation":"\\nYou tried.*"},"level":"error","span":{.*},"children":\[{"message":"the .*","code":null,"level":"help","span":{"file_name":"foo.rs","byte_start":504,"byte_end":516,"line_start":14,"line_end":14,"column_start":0,"column_end":0},"children":\[\]},{"message":"  <u8 as core::ops::Add>","code":null,"level":"help",' foo.log
