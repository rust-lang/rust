include ../tools.mk

PROFILE_DIR=$(TMPDIR)/profiles

all: check_inline check_filter

check_inline:
	$(RUSTC) -O main.rs -Cremark=all -Zremark-dir=$(PROFILE_DIR)
	cat $(PROFILE_DIR)/*.opt.yaml | $(CGREP) -e "inline"
check_filter:
	$(RUSTC) -O main.rs -Cremark=foo -Zremark-dir=$(PROFILE_DIR)
	cat $(PROFILE_DIR)/*.opt.yaml | $(CGREP) -e -v "inline"
