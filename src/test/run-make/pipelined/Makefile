-include ../../run-make-fulldeps/tools.mk

O=$(TMPDIR)/together
S=$(TMPDIR)/separate

all: $O/main $S/main
	# Should produce identical output
	cmp $O/main	$S/main

# Build with combined --emit metadata,link
$O/main: $O/libmiddle.rlib $O/libleaf.rlib
	$(BARE_RUSTC) --out-dir $O main.rs --extern middle=$O/libmiddle.rlib -Ldependency=$O
	$O/main

$O/libmiddle.rlib: middle.rs $O/libleaf.rmeta
	$(BARE_RUSTC) --out-dir $O --emit link middle.rs --extern leaf=$O/libleaf.rmeta

$O/libleaf.rlib $O/libleaf.rmeta: leaf.rs
	$(BARE_RUSTC) --out-dir $O --emit metadata,link leaf.rs

# Build with separate --emit metadata / --emit link
$S/main: $S/libmiddle.rlib $S/libleaf.rlib
	$(BARE_RUSTC) --out-dir $S main.rs --extern middle=$S/libmiddle.rlib -Ldependency=$S
	$S/main

$S/libmiddle.rlib: middle.rs $S/libleaf.rmeta
	$(BARE_RUSTC) --out-dir $S --emit link middle.rs --extern leaf=$S/libleaf.rmeta

$S/libleaf.rlib: leaf.rs
	$(BARE_RUSTC) --out-dir $S --emit link leaf.rs

$S/libleaf.rmeta: leaf.rs
	$(BARE_RUSTC) --out-dir $S --emit metadata -Zmetadata-link leaf.rs
