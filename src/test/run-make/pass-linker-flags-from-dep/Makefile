include ../../run-make-fulldeps/tools.mk

all:
	# Build deps
	$(RUSTC) native_dep_1.rs --crate-type=staticlib
	$(RUSTC) native_dep_2.rs --crate-type=staticlib
	$(RUSTC) rust_dep.rs -l static:-bundle=native_dep_1 -l link-arg=some_flag -l static:-bundle=native_dep_2 --crate-type=lib -Z unstable-options

	# Check sequence of linker args
	$(RUSTC) main.rs --extern lib=$(TMPDIR)/librust_dep.rlib --crate-type=bin --print link-args | $(CGREP) -e 'native_dep_1.*some_flag.*native_dep_2'
