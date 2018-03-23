-include ../tools.mk

all:
	$(RUSTC) debug.rs -C debug-assertions=no
	$(call RUN,debug) good
	$(RUSTC) debug.rs -C opt-level=0
	$(call RUN,debug) bad
	$(RUSTC) debug.rs -C opt-level=1
	$(call RUN,debug) good
	$(RUSTC) debug.rs -C opt-level=2
	$(call RUN,debug) good
	$(RUSTC) debug.rs -C opt-level=3
	$(call RUN,debug) good
	$(RUSTC) debug.rs -C opt-level=s
	$(call RUN,debug) good
	$(RUSTC) debug.rs -C opt-level=z
	$(call RUN,debug) good
	$(RUSTC) debug.rs -O
	$(call RUN,debug) good
	$(RUSTC) debug.rs
	$(call RUN,debug) bad
	$(RUSTC) debug.rs -C debug-assertions=yes -O
	$(call RUN,debug) bad
	$(RUSTC) debug.rs -C debug-assertions=yes -C opt-level=1
	$(call RUN,debug) bad
