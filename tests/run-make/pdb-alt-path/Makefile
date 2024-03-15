include ../tools.mk

# only-windows-msvc

all:
	# Test that we don't have the full path to the PDB file in the binary
	$(RUSTC) main.rs -g --crate-name my_crate_name --crate-type bin -Cforce-frame-pointers
	$(CGREP) "my_crate_name.pdb" < $(TMPDIR)/my_crate_name.exe
	$(CGREP) -v "\\my_crate_name.pdb" < $(TMPDIR)/my_crate_name.exe

	# Test that backtraces still can find debuginfo by checking that they contain symbol names and
	# source locations.
	$(TMPDIR)/my_crate_name.exe &> $(TMPDIR)/backtrace.txt
	$(CGREP) "my_crate_name::fn_in_backtrace" < $(TMPDIR)/backtrace.txt
	$(CGREP) "main.rs:15" < $(TMPDIR)/backtrace.txt

	# Test that explicitly passed `-Clink-arg=/PDBALTPATH:...` is respected
	$(RUSTC) main.rs -g --crate-name my_crate_name --crate-type bin -Clink-arg=/PDBALTPATH:abcdefg.pdb -Cforce-frame-pointers
	$(CGREP) "abcdefg.pdb" < $(TMPDIR)/my_crate_name.exe
	$(CGREP) -v "my_crate_name.pdb" < $(TMPDIR)/my_crate_name.exe
