-include ../tools.mk

# Test for #39529.
# `-z text` causes ld to error if there are any non-PIC sections

ifeq ($(UNAME),Darwin)
all:
else ifdef IS_WINDOWS
all:
else
all:
	$(RUSTC) hello.rs -C link-args=-Wl,-z,text
endif
