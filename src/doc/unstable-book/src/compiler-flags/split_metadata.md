## `split-metadata`

This option instructs `rustc` to only include a stub metadata section in `rlib` and `dylib` crate types instead of full metadata, to reduce their size on disk. You will probably want to combine this option with `--emit=metadata` to produce the full metadata into a separate `.rmeta` file.
