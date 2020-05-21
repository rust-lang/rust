# LSP Extensions

This document describes LSP extensions used by rust-analyzer.
It's a best effort document, when in doubt, consult the source (and send a PR with clarification ;-) ).
We aim to upstream all non Rust-specific extensions to the protocol, but this is not a top priority.
All capabilities are enabled via `experimental` field of `ClientCapabilities`.

## `SnippetTextEdit`

**Client Capability:** `{ "snippetTextEdit": boolean }`

If this capability is set, `WorkspaceEdit`s returned from `codeAction` requests might contain `SnippetTextEdit`s instead of usual `TextEdit`s:

```typescript
interface SnippetTextEdit extends TextEdit {
    insertTextFormat?: InsertTextFormat;
}
```

```typescript
export interface TextDocumentEdit {
	textDocument: VersionedTextDocumentIdentifier;
	edits: (TextEdit | SnippetTextEdit)[];
}
```

When applying such code action, the editor should insert snippet, with tab stops and placeholder.
At the moment, rust-analyzer guarantees that only a single edit will have `InsertTextFormat.Snippet`.

### Example

"Add `derive`" code action transforms `struct S;` into `#[derive($0)] struct S;`

### Unresolved Questions

* Where exactly are `SnippetTextEdit`s allowed (only in code actions at the moment)?
* Can snippets span multiple files (so far, no)?

## `joinLines`

**Server Capability:** `{ "joinLines": boolean }`

This request is send from client to server to handle "Join Lines" editor action.

**Method:** `experimental/JoinLines`

**Request:**

```typescript
interface JoinLinesParams {
    textDocument: TextDocumentIdentifier,
    /// Currently active selections/cursor offsets.
    /// This is an array to support multiple cursors.
    ranges: Range[],
}
```

**Response:**

```typescript
TextEdit[]
```

### Example

```rust
fn main() {
    /*cursor here*/let x = {
        92
    };
}
```

`experimental/joinLines` yields (curly braces are automagiacally removed)

```rust
fn main() {
    let x = 92;
}
```

### Unresolved Question

* What is the position of the cursor after `joinLines`?
  Currently this is left to editor's discretion, but it might be useful to specify on the server via snippets.
  However, it then becomes unclear how it works with multi cursor.

## Structural Search Replace (SSR)

**Server Capability:** `{ "ssr": boolean }`

This request is send from client to server to handle structural search replace -- automated syntax tree based transformation of the source.

**Method:** `experimental/ssr`

**Request:**

```typescript
interface SsrParams {
    /// Search query.
    /// The specific syntax is specified outside of the protocol.
    query: string,
    /// If true, only check the syntax of the query and don't compute the actual edit.
    parseOnly: bool,
}
```

**Response:**

```typescript
WorkspaceEdit
```

### Example

SSR with query `foo($a:expr, $b:expr) ==>> ($a).foo($b)` will transform, eg `foo(y + 5, z)` into `(y + 5).foo(z)`.

### Unresolved Question

* Probably needs search without replace mode
* Needs a way to limit the scope to certain files.
