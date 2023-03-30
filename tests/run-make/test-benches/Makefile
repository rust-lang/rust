include ../tools.mk

# ignore-cross-compile

all:
	# Smoke-test that `#[bench]` isn't entirely broken.
	$(RUSTC) --test smokebench.rs -O
	$(call RUN,smokebench --bench)
	$(call RUN,smokebench --bench noiter)
	$(call RUN,smokebench --bench yesiter)
	$(call RUN,smokebench)
