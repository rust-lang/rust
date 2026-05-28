# `no-steal-thir`

By default, to save on memory, the THIR body (obtained from the `tcx.thir_body` query) is stolen
once no longer used. This is inconvenient for authors of rustc drivers who want to access the THIR.

This option disables the stealing. This has no observable effect on compiler behavior, only on
memory usage.
