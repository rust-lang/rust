# `dump-mono-stats`

--------------------

The `-Z dump-mono-stats` compiler flag generates a file with a list of the monomorphized items in the current crate.
It is useful for investigating compile times.

It accepts an optional directory where the file will be located. If no directory is specified, the file will be placed in the current directory.

See also `-Z dump-mono-stats-format` and `-Z print-mono-items`. Unlike `print-mono-items`,
`dump-mono-stats` aggregates monomorphized items by definition and includes a size estimate of how
large the item is when codegened.

See <https://rustc-dev-guide.rust-lang.org/backend/monomorph.html> for an overview of monomorphized items.
