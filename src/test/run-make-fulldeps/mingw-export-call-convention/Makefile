include ../tools.mk

# only-windows-gnu

all:
	$(RUSTC) foo.rs
	# FIXME: we should make sure __stdcall calling convention is used here
	# but that only works with LLD right now
	nm -g "$(call IMPLIB,foo)" | $(CGREP) bar
