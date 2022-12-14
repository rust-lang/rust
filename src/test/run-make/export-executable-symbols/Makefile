include ../../run-make-fulldeps/tools.mk

# ignore-wasm32
# ignore-wasm64
# ignore-none no-std is not supported
# only-linux

all:
	$(RUSTC) -Zexport-executable-symbols  main.rs --target $(TARGET) --crate-type=bin
	nm $(TMPDIR)/main | $(CGREP) exported_symbol

