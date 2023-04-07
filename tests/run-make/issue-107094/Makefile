# needs-git-hash

include ../tools.mk

all:
	$(BARE_RUSTC) --version --verbose | $(CGREP) -i -e "commit-hash: [0-9a-f]{40}" "commit-date: [0-9\-]+"
	$(BARE_RUSTDOC) --version --verbose | $(CGREP) -i -e "commit-hash: [0-9a-f]{40}" "commit-date: [0-9\-]+"
