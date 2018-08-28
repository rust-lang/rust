-include ../tools.mk

all:
	$(RUSTC) foo.rs -C target-cpu=native 2>&1 | tee $(TMPDIR)/out.log
	$(call RUN,foo)
	# Make sure no warnings about "unknown CPU `native`" were emitted
	if [ "$$(wc -c $(TMPDIR)/out.log | cut -d' ' -f 1)" = "0" ]; then \
	  echo no warnings generated; \
	else \
	  exit 1; \
	fi
