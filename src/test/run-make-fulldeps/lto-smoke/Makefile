-include ../tools.mk

all: noparam bool_true bool_false thin fat

noparam:
	$(RUSTC) lib.rs
	$(RUSTC) main.rs -C lto
	$(call RUN,main)

bool_true:
	$(RUSTC) lib.rs
	$(RUSTC) main.rs -C lto=yes
	$(call RUN,main)


bool_false:
	$(RUSTC) lib.rs
	$(RUSTC) main.rs -C lto=off
	$(call RUN,main)

thin:
	$(RUSTC) lib.rs
	$(RUSTC) main.rs -C lto=thin
	$(call RUN,main)

fat:
	$(RUSTC) lib.rs
	$(RUSTC) main.rs -C lto=fat
	$(call RUN,main)

