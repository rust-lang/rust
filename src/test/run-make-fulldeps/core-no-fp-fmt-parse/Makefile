-include ../tools.mk

all:
	CARGO_PKG_NAME=core $(RUSTC) --edition=2018 --crate-type=rlib ../../../../library/core/src/lib.rs --cfg no_fp_fmt_parse
