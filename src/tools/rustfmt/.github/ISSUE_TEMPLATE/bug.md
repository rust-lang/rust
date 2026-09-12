---
name: General rustfmt bug Report
about: Create a general bug report for rustfmt. Prefer more specialized issue templates if applicable.
labels: C-bug
---
<!--
Thank you for filing a bug report! 🐛 Please provide a short summary of the bug,
along with any information you feel relevant to replicating the bug.
-->

## Summary

<!--
Please include a reproducer for the bug you are describing.
If possible, try to provide a Minimal, Complete and Verifiable example.
You can read "Rust Bug Minimization Patterns" for how to create smaller examples.
http://blog.pnkfx.org/blog/2019/11/18/rust-bug-minimization-patterns/
-->

I tried to format this code:

```rust
<code>
```

### Expected behavior

I expected to see this happen: *explanation*

### Actual behavior

Instead, this happened: *explanation*


## Configuration

<!--
Include any CLI options used, and include the configuration file (e.g.
`rustfmt.toml`).
-->

`rustfmt` cli options used (if applicable):

```bash
$ <rustfmt_options>
```

`rustfmt` configuration file (e.g. `rustfmt.toml`, if applicable):

```md
<configuration_file>
```


## Reproduction Steps

<!-- Include any steps that might be needed to reproduce this behavior -->

1. ...


## Meta
<!--
If you're using the stable version of rustfmt, you should also check if the
bug also exists in the beta or nightly versions.
-->

`rustfmt --version`:
```
<version>
```
