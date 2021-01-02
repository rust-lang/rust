include ../../run-make-fulldeps/tools.mk

# Tests that the dep graph contains printed def paths,
# instead of just hashes

all: main.rs
	cp main.rs $(TMPDIR)/
	cd $(TMPDIR)
	$(RUSTC) -Z dump-dep-graph $<
	# Make sure that we can find a local function
	# (my_function) and something from a foreign crate
	# (compiler_builtins) in the dep graph
	grep my_function dep_graph.txt
	grep compiler_builtins dep_graph.txt
	rm dep_graph.txt dep_graph.dot
