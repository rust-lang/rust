include ../../run-make-fulldeps/tools.mk

# Different optimization levels imply different values for `-Zshare-generics`,
# so try out a whole bunch of combinations to make sure everything is compatible
all:
	# First up, try some defaults
	$(RUSTC) --crate-type rlib foo.rs
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=3

	# Next try mixing up some things explicitly
	$(RUSTC) --crate-type rlib foo.rs -Z share-generics=no
	$(RUSTC) --crate-type dylib bar.rs -Z share-generics=no
	$(RUSTC) --crate-type rlib foo.rs -Z share-generics=no
	$(RUSTC) --crate-type dylib bar.rs -Z share-generics=yes
	$(RUSTC) --crate-type rlib foo.rs -Z share-generics=yes
	$(RUSTC) --crate-type dylib bar.rs -Z share-generics=no
	$(RUSTC) --crate-type rlib foo.rs -Z share-generics=yes
	$(RUSTC) --crate-type dylib bar.rs -Z share-generics=yes

	# Now combine a whole bunch of options together
	$(RUSTC) --crate-type rlib foo.rs
	$(RUSTC) --crate-type dylib bar.rs
	$(RUSTC) --crate-type dylib bar.rs -Z share-generics=no
	$(RUSTC) --crate-type dylib bar.rs -Z share-generics=yes
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=1
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=1 -Z share-generics=no
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=1 -Z share-generics=yes
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=2
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=2 -Z share-generics=no
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=2 -Z share-generics=yes
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=3
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=3 -Z share-generics=no
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=3 -Z share-generics=yes
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=s
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=s -Z share-generics=no
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=s -Z share-generics=yes
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=z
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=z -Z share-generics=no
	$(RUSTC) --crate-type dylib bar.rs -C opt-level=z -Z share-generics=yes
