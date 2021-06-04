-include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTDOC) --output-format=json x.html 2>&1 | diff - output-format-json.stderr
	$(RUSTC) --force-warns dead_code x.rs 2>&1 | diff - force-warns.stderr
