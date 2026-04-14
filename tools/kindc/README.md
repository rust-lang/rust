# kindc — ThingOS Kind Schema Compiler

`kindc` is the canonical compiler for ThingOS Kind schemas. it parses `.kind` source files, validates them into a resolved intermediate representation, and generates deterministic Rust code.

## Language

The `.kind` language is a simple, declarative DSL for defining system types.

### Syntax Example

```kind
/// A logical or physical location
kind thingos.place = struct {
  name: string,
  kind: thingos.place.kind,
}

kind thingos.place.kind = enum {
  System,
  Realm,
  Device,
  Remote,
}
```

## Features

- **Dotted Names**: All kinds have a canonical global name (e.g., `thingos.person`).
- **Structs**: Fixed-layout data structures.
- **Enums**: Tagged unions with optional unit, tuple, or struct payloads.
- **Type Aliases**: Create aliases for existing types (e.g., `kind byte = u8`).
- **Generics**: Built-in containers like `list<T>`, `option<T>`, and `result<T, E>`.
- **References**: `ref<T>` represents a 128-bit `ThingId` handle to a thing of kind `T`.
- **Doc Comments**: Triple-slash `///` comments are propagated to generated code.
- **KindId**: Every kind gets a stable 128-bit `KindId` derived from its name and structural shape (blake3 hash).

## Usage

### CLI

```bash
cargo run -p kindc -- <INPUT_DIR_OR_FILE> -o <OUTPUT_DIR>
```

### Just Recipes

- `just kindc <args>`: Run the compiler.
- `just kindc-gen`: Regenerate the canonical system schema Rust output in `tools/kindc/fixtures/generated`.

## Pipeline

1. **Parse**: Source files are parsed into an untyped Abstract Syntax Tree (AST).
2. **Resolve**: Names are resolved across files, and built-ins are linked.
3. **Validate**: Semantic checks are performed (e.g., circular struct references, arity mismatches).
4. **Generate**: Rust code is emitted based on the resolved IR.
