include ../tools.mk

all:
	$(RUSTC) --edition=2021 --crate-type=rlib ../../../../library/core/src/lib.rs --cfg no_fp_fmt_parse
