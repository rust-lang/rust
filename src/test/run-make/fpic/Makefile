-include ../tools.mk

# Test for #39529.
# `-z text` causes ld to error if there are any non-PIC sections

ifeq ($(UNAME),Darwin)
all:
else ifdef IS_WINDOWS
all:
else ifeq ($(findstring stage0,$(RUST_BUILD_STAGE)),stage0)
all:
	$(RUSTC) hello.rs -C link-args=-Wl,-z,text
else
all:
	$(RUSTC) hello.rs -C link-args='-z text'
endif
