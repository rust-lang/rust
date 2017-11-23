-include ../tools.mk

all: default
	$(RUSTC) --target x86_64-pc-windows-gnu --print cfg | $(CGREP) windows
	$(RUSTC) --target x86_64-pc-windows-gnu --print cfg | $(CGREP) x86_64
	$(RUSTC) --target i686-pc-windows-msvc --print cfg | $(CGREP) msvc
	$(RUSTC) --target i686-apple-darwin --print cfg | $(CGREP) macos
	$(RUSTC) --target i686-unknown-linux-gnu --print cfg | $(CGREP) gnu

ifdef IS_WINDOWS
default:
	$(RUSTC) --print cfg | $(CGREP) windows
else
default:
	$(RUSTC) --print cfg | $(CGREP) unix
endif
