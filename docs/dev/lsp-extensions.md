# LSP Extensions

This document describes LSP extensions used by rust-analyzer.
It's a best effort document, when in doubt, consult the source (and send a PR with clarification ;-) ).
We aim to upstream all non Rust-specific extensions to the protocol, but this is not a top priority.
All capabilities are enabled via `experimental` field of `ClientCapabilities` or `ServerCapabilities`.
Requests which we hope to upstream live under `experimental/` namespace.
Requests, which are likely to always remain specific to `rust-analyzer` are under `rust-analyzer/` namespace.

## Snippet `TextEdit`

**Issue:** https://github.com/microsoft/language-server-protocol/issues/724

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

## `CodeAction` Groups

**Issue:** https://github.com/microsoft/language-server-protocol/issues/994

**Client Capability:** `{ "codeActionGroup": boolean }`

If this capability is set, `CodeAction` returned from the server contain an additional field, `group`:

```typescript
interface CodeAction {
    title: string;
    group?: string;
    ...
}
```

All code-actions with the same `group` should be grouped under single (extendable) entry in lightbulb menu.
The set of actions `[ { title: "foo" }, { group: "frobnicate", title: "bar" }, { group: "frobnicate", title: "baz" }]` should be rendered as

```
💡
  +-------------+
  | foo         |
  +-------------+-----+
  | frobnicate >| bar |
  +-------------+-----+
                | baz |
                +-----+
```

Alternatively, selecting `frobnicate` could present a user with an additional menu to choose between `bar` and `baz`.

### Example

```rust
fn main() {
    let x: Entry/*cursor here*/ = todo!();
}
```

Invoking code action at this position will yield two code actions for importing `Entry` from either `collections::HashMap` or `collection::BTreeMap`, grouped under a single "import" group.

### Unresolved Questions

* Is a fixed two-level structure enough?
* Should we devise a general way to encode custom interaction protocols for GUI refactorings?

## Join Lines

**Issue:** https://github.com/microsoft/language-server-protocol/issues/992

**Server Capability:** `{ "joinLines": boolean }`

This request is send from client to server to handle "Join Lines" editor action.

**Method:** `experimental/joinLines`

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

## Matching Brace

**Issue:** https://github.com/microsoft/language-server-protocol/issues/999

**Server Capability:** `{ "matchingBrace": boolean }`

This request is send from client to server to handle "Matching Brace" editor action.

**Method:** `experimental/matchingBrace`

**Request:**

```typescript
interface MatchingBraceParams {
    textDocument: TextDocumentIdentifier,
    /// Position for each cursor
    positions: Position[],
}
```

**Response:**

```typescript
Position[]
```

### Example

```rust
fn main() {
    let x: Vec<()>/*cursor here*/ = vec![]
}
```

`experimental/matchingBrace` yields the position of `<`.
In many cases, matching braces can be handled by the editor.
However, some cases (like disambiguating between generics and comparison operations) need a real parser.
Moreover, it would be cool if editors didn't need to implement even basic language parsing

### Unresolved Question

* Should we return a a nested brace structure, to allow paredit-like actions of jump *out* of the current brace pair?
  This is how `SelectionRange` request works.
* Alternatively, should we perhaps flag certain `SelectionRange`s as being brace pairs?

## Analyzer Status

**Method:** `rust-analyzer/analyzerStatus`

**Request:** `null`

**Response:** `string`

Returns internal status message, mostly for debugging purposes.

## Collect Garbage

**Method:** `rust-analyzer/collectGarbage`

**Request:** `null`

**Response:** `null`

Frees some caches. For internal use, and is mostly broken at the moment.

## Syntax Tree

**Method:** `rust-analyzer/syntaxTree`

**Request:**

```typescript
interface SyntaxTeeParams {
    textDocument: TextDocumentIdentifier,
    range?: Range,
}
```

**Response:** `string`

Returns textual representation of a parse tree for the file/selected region.
Primarily for debugging, but very useful for all people working on rust-analyzer itself.

## Expand Macro

**Method:** `rust-analyzer/expandMacro`

**Request:**

```typescript
interface ExpandMacroParams {
    textDocument: TextDocumentIdentifier,
    position?: Position,
}
```

**Response:**

```typescript
interface ExpandedMacro {
    name: string,
    expansion: string,
}
```

Expands macro call at a given position.
