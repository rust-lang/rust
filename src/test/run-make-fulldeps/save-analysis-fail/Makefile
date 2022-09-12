include ../tools.mk
all: code
krate2: krate2.rs
	$(RUSTC) $<
code: foo.rs krate2
	$(RUSTC) foo.rs -Zsave-analysis || exit 0
