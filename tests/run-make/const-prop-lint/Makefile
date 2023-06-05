include ../tools.mk

# Test that emitting an error because of arithmetic
# overflow lint does not leave .o files around
# because of interrupted codegen.

all:
	$(RUSTC) input.rs; test $$? -eq 1
	ls *.o; test $$? -ne 0
