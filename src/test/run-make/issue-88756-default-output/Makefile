include ../../run-make-fulldeps/tools.mk

all:
	$(BARE_RUSTDOC) 2>&1 | sed -E 's@/nightly/|/beta/|/stable/|/1\.[0-9]+\.[0-9]+/@/$$CHANNEL/@g' | diff - output-default.stdout
