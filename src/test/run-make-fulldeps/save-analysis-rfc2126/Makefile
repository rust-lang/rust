-include ../tools.mk

all: extern_absolute_paths.rs extern_in_paths.rs krate2
	$(RUSTC) extern_absolute_paths.rs -Zsave-analysis
	$(RUSTC) extern_in_paths.rs -Zsave-analysis

krate2: krate2.rs
	$(RUSTC) $<
