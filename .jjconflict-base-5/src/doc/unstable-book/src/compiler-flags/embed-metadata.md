## `embed-metadata`

This option instructs `rustc` to include the full metadata in `rlib` and `dylib` crate types. The default value is `yes` (enabled). If disabled (`no`), only stub metadata will be stored in these files, to reduce their size on disk. When using `-Zembed-metadata=no`, you will probably want to use `--emit=metadata` to produce the full metadata into a separate `.rmeta` file.
