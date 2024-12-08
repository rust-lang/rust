# Settings Example

Add the following to settings.json to mark Rust library sources as read-only:

```json
"files.readonlyInclude": {
  "**/.cargo/registry/src/**/*.rs": true,
  "**/lib/rustlib/src/rust/library/**/*.rs": true,
},
```
