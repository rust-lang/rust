# `split-dwarf-out-dir`

On systems which use DWARF debug info this flag causes `.dwo` files produced
by `-C split-debuginfo` to be written to the specified directory rather than
placed next to the object files. This is mostly useful if you have a build
system which needs to control where to find compile outputs without running the
compiler and have to put your `.dwo` files in a separate directory.
