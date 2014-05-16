RUSTC ?= rustc
RUSTC_FLAGS ?=

check: validator-build
	./target/validator content/**/*.md

validator-build: validator/bin.rs
	$(RUSTC) validator/bin.rs --output-dir target

validator-clean:
	@rm -rf target

doc-build:
	@nanoc

doc-server:
	@nanoc autocompile -p 3002 -H thin

doc-clean:
	@rm -rf ./tmp
	@rm -rf ./public
	@rm crash.log

clean: validator-clean doc-clean

.PHONY: validator-clean doc-clean doc-build doc-server