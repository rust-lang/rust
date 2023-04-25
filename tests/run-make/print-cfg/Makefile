# needs-llvm-components: x86 arm

include ../tools.mk

all: default output_to_file
	$(RUSTC) --target x86_64-pc-windows-gnu --print cfg | $(CGREP) windows
	$(RUSTC) --target x86_64-pc-windows-gnu --print cfg | $(CGREP) x86_64
	$(RUSTC) --target i686-pc-windows-msvc --print cfg | $(CGREP) msvc
	$(RUSTC) --target i686-apple-darwin --print cfg | $(CGREP) macos
	$(RUSTC) --target i686-unknown-linux-gnu --print cfg | $(CGREP) gnu
	$(RUSTC) --target arm-unknown-linux-gnueabihf --print cfg | $(CGREP) target_abi=
	$(RUSTC) --target arm-unknown-linux-gnueabihf --print cfg | $(CGREP) eabihf

output_to_file:
	-rm $(TMPDIR)/cfg.txt
	$(RUSTC) --target x86_64-pc-windows-gnu --print=cfg -o $(TMPDIR)/cfg.txt
	$(CGREP) windows < $(TMPDIR)/cfg.txt
	-rm $(TMPDIR)/cfg.txt
	$(RUSTC) --target x86_64-pc-windows-gnu --print=target-cpus -o $(TMPDIR)/cfg.txt
	$(CGREP) native < $(TMPDIR)/cfg.txt

ifdef IS_WINDOWS
default:
	$(RUSTC) --print cfg | $(CGREP) windows
else
default:
	$(RUSTC) --print cfg | $(CGREP) unix
endif
