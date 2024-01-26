# ignore-msvc

include ../tools.mk

all:
	$(RUSTC) depa.rs
	$(RUSTC) depb.rs
	$(RUSTC) depc.rs
	$(RUSTC) empty.rs --cfg bar --print=link-args | $(CGREP) '"-ltesta" "-ltestb" "-ltesta"'
	$(RUSTC) empty.rs --print=link-args | $(CGREP) '"-ltesta"'
	$(RUSTC) empty.rs --print=link-args | $(CGREP) -v '"-ltestb"'
	$(RUSTC) empty.rs --print=link-args | $(CGREP) -v '"-ltesta" "-ltesta" "-ltesta"'
