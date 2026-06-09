# `temps-dir`

--------------------

The `-Ztemps-dir` compiler flag specifies the directory to write the
intermediate files in. If not set, the output directory is used. This option is
useful if you are running more than one instance of `rustc` (e.g. with different
`--crate-type` settings), and you need to make sure they are not overwriting
each other's intermediate files. No files are kept unless `-C save-temps=yes` is
also set.
