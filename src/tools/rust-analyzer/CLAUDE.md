**Reminder: All AI usage must be disclosed in commit messages, see
CONTRIBUTING.md for more details.**

## Build Commands

```bash
cargo build                    # Build all crates
cargo test                     # Run all tests
cargo test -p <crate>          # Run tests for a specific crate (e.g., cargo test -p hir-ty)
cargo lint                     # Run clippy on all targets
cargo xtask codegen            # Run code generation
cargo xtask tidy               # Run tidy checks
UPDATE_EXPECT=1 cargo test     # Update test expectations (snapshot tests)
RUN_SLOW_TESTS=1 cargo test    # Run heavy/slow tests
```

## Key Architectural Invariants

- Typing in a function body never invalidates global derived data
- Parser/syntax tree is built per-file to enable parallel parsing
- The server is stateless (HTTP-like); context must be re-created from request parameters
- Cancellation uses salsa's cancellation mechanism; computations panic with a `Cancelled` payload

### Code Generation

Generated code is committed to the repo. Grammar and AST are generated from `ungrammar`. Run `cargo test -p xtask` after adding inline parser tests (`// test test_name` comments).

## Testing

Tests are snapshot-based using `expect-test`. Test fixtures use a mini-language:
- `$0` marks cursor position
- `// ^^^^` labels attach to the line above
- `//- minicore: sized, fn` includes parts of minicore (minimal core library)
- `//- /path/to/file.rs crate:name deps:dep1,dep2` declares files/crates

## Style Notes

- Use `stdx::never!` and `stdx::always!` instead of `assert!` for recoverable invariants
- Use `T![fn]` macro instead of `SyntaxKind::FN_KW`
- Use keyword name mangling over underscore prefixing for identifiers: `crate` → `krate`, `fn` → `func`, `struct` → `strukt`, `type` → `ty`
