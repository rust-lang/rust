-include ../tools.mk

# only-linux
# only-x86_64
#
# I *really* don't want to deal with a cross-platform way to compare file sizes,
# tests in `make` sort of are awful

all: $(TMPDIR)/out.log
	# Make sure no warnings about "unknown CPU `native`" were emitted
	if [ "$$(wc -c $(TMPDIR)/out.log | cut -d' ' -f 1)" = "0" ]; then \
	  echo no warnings generated; \
	else \
	  exit 1; \
	fi


$(TMPDIR)/out.log:
	$(RUSTC) foo.rs -C target-cpu=native 2>&1 | tee $(TMPDIR)/out.log
	$(call RUN,foo)
