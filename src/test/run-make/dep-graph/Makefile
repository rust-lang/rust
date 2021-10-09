-include ../../run-make-fulldeps/tools.mk

# ignore-cross-compile

# Just verify that we successfully run and produce dep graphs when requested.

all:
	RUST_DEP_GRAPH=$(TMPDIR)/dep-graph $(RUSTC) \
        -Cincremental=$(TMPDIR)/incr \
        -Zquery-dep-graph -Zdump-dep-graph foo.rs
	test -f $(TMPDIR)/dep-graph.txt
	test -f $(TMPDIR)/dep-graph.dot
