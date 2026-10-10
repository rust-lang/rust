# `SDKROOT`

This environment variable is used on Apple targets.
It is passed through to the linker (currently either directly or via the `-syslibroot` flag).

Note that this variable is not always respected. When the SDKROOT is clearly wrong (e.g. when the platform of the SDK does not match the `--target` used by rustc), this is ignored and rustc does its own detection.
