-include ../../run-make-fulldeps/tools.mk

O=$(TMPDIR)

all: $O/main
	$O/main

$O/main: $O/libmiddle.rlib $O/libleaf.rlib
	$(RUSTC) main.rs --extern middle=$O/libmiddle.rlib -Ldependency=$O

$O/libmiddle.rlib: middle.rs $O/libleaf.rmeta
	$(RUSTC) --emit link middle.rs --extern leaf=$O/libleaf.rmeta

$O/libleaf.rlib $O/libleaf.rmeta: leaf.rs
	$(RUSTC) --emit metadata,link leaf.rs
