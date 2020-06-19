-include ../../run-make-fulldeps/tools.mk

# only-x86_64-unknown-linux-musl

# How to manually run this
# $ ./x.py test --target x86_64-unknown-linux-musl src/test/run-make/static-pie

all:
	$(RUSTC) --target $(TARGET) -C target-feature=+crt-static test-aslr.rs
	# Check that no dynamic interpreter is set
	! readelf -l $(call RUN_BINFILE,test-aslr) | $(CGREP) INTERP
	# Check that we have a dynamic executable
	readelf -l $(call RUN_BINFILE,test-aslr) | $(CGREP) DYNAMIC
	# Check for address space layout randomization
	$(call RUN,test-aslr) --test-aslr
