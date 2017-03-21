-include ../tools.mk
all: code
krate2: krate2.rs
	$(RUSTC) $<
code: foo.rs krate2 sugar.rs
	$(RUSTC) foo.rs -Zsave-analysis-csv
	$(RUSTC) foo.rs -Zsave-analysis
	$(RUSTC) foo.rs -Zsave-analysis-api
	$(RUSTC) sugar.rs -Zsave-analysis-csv
