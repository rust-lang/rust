-include ../tools.mk

# Test for #39529. 
# `-z text` causes ld to error if there are any non-PIC sections

all:
ifndef IS_WINDOWS
	$(RUSTC) hello.rs -C link-args=-Wl,-z,text
endif
