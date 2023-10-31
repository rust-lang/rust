# ignore-cross-compile
include ../tools.mk

all:
	$(RUSTC) success.rs; [ $$? -eq 0 ]
	$(RUSTC) --invalid-arg-foo; [ $$? -eq 1 ]
	$(RUSTC) compile-error.rs; [ $$? -eq 1 ]
	RUSTC_ICE=0 $(RUSTC) -Ztreat-err-as-bug compile-error.rs; [ $$? -eq 101 ]
	$(RUSTDOC) -o $(TMPDIR)/exit-code success.rs; [ $$? -eq 0 ]
	$(RUSTDOC) --invalid-arg-foo; [ $$? -eq 1 ]
	$(RUSTDOC) compile-error.rs; [ $$? -eq 1 ]
	$(RUSTDOC) lint-failure.rs; [ $$? -eq 1 ]
