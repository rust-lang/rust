# only-linux

# Ensure that the enriched `gcc` linker-flavor passes the requested linker to `cc`

-include ../../run-make-fulldeps/tools.mk
RUSTC_FLAGS = -C linker-flavor=gcc:lld -Z unstable-options --print link-args

all:
	$(RUSTC) $(RUSTC_FLAGS) empty.rs | $(CGREP) "fuse-ld=lld"
