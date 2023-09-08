include ../tools.mk

# only-windows-msvc

# tests if the pdb contains the following information in the LF_BUILDINFO:
# 1. the commandline args to compile it (cmd)
# 2. full path to the compiler (cl)

# we just do a stringsearch on the pdb, as these need to show up at least once, as the LF_BUILDINFO is created for each cgu
# actual parsing would be better, but this is a simple and good enough solution for now

all:
	$(RUSTC_ORIGINAL) main.rs -g --crate-name my_crate_name --crate-type bin -C metadata=dc9ef878b0a48666 --out-dir $(TMPDIR)
	cat '$(TMPDIR)/my_crate_name.pdb' | grep -F '$(RUSTC_ORIGINAL)'
# using a file containing the string so I don't have problems with escaping quotes and spaces 
	cat '$(TMPDIR)/my_crate_name.pdb' | grep -f 'stringlist.txt'
