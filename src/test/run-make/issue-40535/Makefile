-include ../tools.mk

# The ICE occurred in the following situation:
# * `foo` declares `extern crate bar, baz`, depends only on `bar` (forgetting `baz` in `Cargo.toml`)
# * `bar` declares and depends on `extern crate baz`
# * All crates built in metadata-only mode (`cargo check`)
all:
	# cc https://github.com/rust-lang/rust/issues/40623
	$(RUSTC) baz.rs --emit=metadata
	$(RUSTC) bar.rs --emit=metadata --extern baz=$(TMPDIR)/libbaz.rmeta
	$(RUSTC) foo.rs --emit=metadata --extern bar=$(TMPDIR)/libbar.rmeta 2>&1 | \
		$(CGREP) -v "unexpectedly panicked"
	# ^ Succeeds if it doesn't find the ICE message
