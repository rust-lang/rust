# ignore-msvc

include ../tools.mk

all:
	$(RUSTC) depa.rs
	$(RUSTC) depb.rs
	$(RUSTC) depc.rs
	$(RUSTC) empty.rs --cfg bar 2>&1 | $(CGREP) '"-ltesta" "-ltestb" "-ltesta"'
	$(RUSTC) empty.rs 2>&1 | $(CGREP) '"-ltesta"'
	$(RUSTC) empty.rs 2>&1 | $(CGREP) -v '"-ltestb"'
	$(RUSTC) empty.rs 2>&1 | $(CGREP) -v '"-ltesta" "-ltesta"'
