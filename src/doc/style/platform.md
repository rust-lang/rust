% FFI and platform-specific code **[FIXME]**

> **[FIXME]** Not sure where this should live.

When writing cross-platform code, group platform-specific code into a
module called `platform`. Avoid `#[cfg]` directives outside this
`platform` module.
