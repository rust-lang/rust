-include ../tools.mk

all: default
	$(RUSTC) --target x86_64-pc-windows-gnu --print cfg | grep windows
	$(RUSTC) --target x86_64-pc-windows-gnu --print cfg | grep x86_64
	$(RUSTC) --target i686-pc-windows-msvc --print cfg | grep msvc
	$(RUSTC) --target i686-apple-darwin --print cfg | grep macos
	$(RUSTC) --target i686-unknown-linux-gnu --print cfg | grep gnu

ifdef IS_WINDOWS
default:
	$(RUSTC) --print cfg | grep windows
else
default:
	$(RUSTC) --print cfg | grep unix
endif
