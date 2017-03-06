-include ../tools.mk

ifdef IS_WINDOWS
# Do nothing on MSVC.
all:
	exit 0
else
all:
	$(RUSTC) -C opt-level=3 --emit=obj used.rs
	nm -C used.o | grep FOO
	nm -C used.o | grep -v BAR
endif
