# `shell-argfiles`

--------------------

The `-Zshell-argfiles` compiler flag allows argfiles to be parsed using POSIX
"shell-style" quoting. When enabled, the compiler will use `shlex` to parse the
arguments from argfiles specified with `@shell:<path>`.

Because this feature controls the parsing of input arguments, the
`-Zshell-argfiles` flag must be present before the argument specifying the
shell-style argument file.
