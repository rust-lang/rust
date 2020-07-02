# LSP Extensions

This document describes LSP extensions used by rust-analyzer.
It's a best effort document, when in doubt, consult the source (and send a PR with clarification ;-) ).
We aim to upstream all non Rust-specific extensions to the protocol, but this is not a top priority.
All capabilities are enabled via `experimental` field of `ClientCapabilities` or `ServerCapabilities`.
Requests which we hope to upstream live under `experimental/` namespace.
Requests, which are likely to always remain specific to `rust-analyzer` are under `rust-analyzer/` namespace.

If you want to be notified about the changes to this document, subscribe to [#4604](https://github.com/rust-analyzer/rust-analyzer/issues/4604).

## `initializationOptions`

As `initializationOptions`, `rust-analyzer` expects `"rust-analyzer"` section of the configuration.
That is, `rust-analyzer` usually sends `"workspace/configuration"` request with `{ "items": ["rust-analyzer"] }` payload.
`initializationOptions` should contain the same data that would be in the first item of the result.
It's OK to not send anything, then all the settings would take their default values.
However, some settings can not be changed after startup at the moment.

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

## Lazy assists with `ResolveCodeAction`

**Issue:** https://github.com/microsoft/language-server-protocol/issues/787

**Client Capability** `{ "resolveCodeAction": boolean }`

If this capability is set, the assists will be computed lazily. Thus `CodeAction` returned from the server will only contain `id` but not `edit` or `command` fields. The only exclusion from the rule is the diagnostic edits.

After the client got the id, it should then call `experimental/resolveCodeAction` command on the server and provide the following payload:

```typescript
interface ResolveCodeActionParams {
    id: string;
    codeActionParams: lc.CodeActionParams;
}
```

As a result of the command call the client will get the respective workspace edit (`lc.WorkspaceEdit`).

### Unresolved Questions

* Apply smarter filtering for ids?
* Upon `resolveCodeAction` command only call the assits which should be resolved and not all of them?

## Parent Module

**Issue:** https://github.com/microsoft/language-server-protocol/issues/1002

**Server Capability:** `{ "parentModule": boolean }`

This request is send from client to server to handle "Goto Parent Module" editor action.

**Method:** `experimental/parentModule`

**Request:** `TextDocumentPositionParams`

**Response:** `Location | Location[] | LocationLink[] | null`


### Example

```rust
// src/main.rs
mod foo;
// src/foo.rs

/* cursor here*/
```

`experimental/parentModule` returns a single `Link` to the `mod foo;` declaration.

### Unresolved Question

* An alternative would be to use a more general "gotoSuper" request, which would work for super methods, super classes and super modules.
  This is the approach IntelliJ Rust is takeing.
  However, experience shows that super module (which generally has a feeling of navigation between files) should be separate.
  If you want super module, but the cursor happens to be inside an overriden function, the behavior with single "gotoSuper" request is surprising.

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

**Response:** `TextEdit[]`

### Example

```rust
fn main() {
    /*cursor here*/let x = {
        92
    };
}
```

`experimental/joinLines` yields (curly braces are automagically removed)

```rust
fn main() {
    let x = 92;
}
```

### Unresolved Question

* What is the position of the cursor after `joinLines`?
  Currently this is left to editor's discretion, but it might be useful to specify on the server via snippets.
  However, it then becomes unclear how it works with multi cursor.

## On Enter

**Issue:** https://github.com/microsoft/language-server-protocol/issues/1001

**Server Capability:** `{ "onEnter": boolean }`

This request is send from client to server to handle <kbd>Enter</kbd> keypress.

**Method:** `experimental/onEnter`

**Request:**: `TextDocumentPositionParams`

**Response:**

```typescript
SnippetTextEdit[]
```

### Example

```rust
fn main() {
    // Some /*cursor here*/ docs
    let x = 92;
}
```

`experimental/onEnter` returns the following snippet

```rust
fn main() {
    // Some
    // $0 docs
    let x = 92;
}
```

The primary goal of `onEnter` is to handle automatic indentation when opening a new line.
This is not yet implemented.
The secondary goal is to handle fixing up syntax, like continuing doc strings and comments, and escaping `\n` in string literals.

As proper cursor positioning is raison-d'etat for `onEnter`, it uses `SnippetTextEdit`.

### Unresolved Question

* How to deal with synchronicity of the request?
  One option is to require the client to block until the server returns the response.
  Another option is to do a OT-style merging of edits from client and server.
  A third option is to do a record-replay: client applies heuristic on enter immediatelly, then applies all user's keypresses.
  When the server is ready with the response, the client rollbacks all the changes and applies the recorded actions on top of the correct response.
* How to deal with multiple carets?
* Should we extend this to arbitrary typed events and not just `onEnter`?

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

## Runnables

**Issue:** https://github.com/microsoft/language-server-protocol/issues/944

**Server Capability:** `{ "runnables": { "kinds": string[] } }`

This request is send from client to server to get the list of things that can be run (tests, binaries, `cargo check -p`).

**Method:** `experimental/runnables`

**Request:**

```typescript
interface RunnablesParams {
    textDocument: TextDocumentIdentifier;
    /// If null, compute runnables for the whole file.
    position?: Position;
}
```

**Response:** `Runnable[]`

```typescript
interface Runnable {
    label: string;
    /// If this Runnable is associated with a specific function/module, etc, the location of this item
    location?: LocationLink;
    /// Running things is necessary technology specific, `kind` needs to be advertised via server capabilities,
    // the type of `args` is specific to `kind`. The actual running is handled by the client.
    kind: string;
    args: any;
}
```

rust-analyzer supports only one `kind`, `"cargo"`. The `args` for `"cargo"` look like this:

```typescript
{
    workspaceRoot?: string;
    cargoArgs: string[];
    executableArgs: string[];
}
```

## Analyzer Status

**Method:** `rust-analyzer/analyzerStatus`

**Request:** `null`

**Response:** `string`

Returns internal status message, mostly for debugging purposes.

## Reload Workspace

**Method:** `rust-analyzer/reloadWorkspace`

**Request:** `null`

**Response:** `null`

Reloads project information (that is, re-executes `cargo metadata`).

## Status Notification

**Client Capability:** `{ "statusNotification": boolean }`

**Method:** `rust-analyzer/status`

**Notification:** `"loading" | "ready" | "invalid" | "needsReload"`

This notification is sent from server to client.
The client can use it to display persistent status to the user (in modline).
For `needsReload` state, the client can provide a context-menu action to run `rust-analyzer/reloadWorkspace` request.

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
    position: Position,
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

## Inlay Hints

**Method:** `rust-analyzer/inlayHints`

This request is send from client to server to render "inlay hints" -- virtual text inserted into editor to show things like inferred types.
Generally, the client should re-query inlay hints after every modification.
Note that we plan to move this request to `experimental/inlayHints`, as it is not really Rust-specific, but the current API is not necessary the right one.
Upstream issue: https://github.com/microsoft/language-server-protocol/issues/956

**Request:**

```typescript
interface InlayHintsParams {
    textDocument: TextDocumentIdentifier,
}
```

**Response:** `InlayHint[]`

```typescript
interface InlayHint {
    kind: "TypeHint" | "ParameterHint" | "ChainingHint",
    range: Range,
    label: string,
}
```

## Hover Actions

**Client Capability:** `{ "hoverActions": boolean }`

If this capability is set, `Hover` request returned from the server might contain an additional field, `actions`:

```typescript
interface Hover {
    ...
    actions?: CommandLinkGroup[];
}

interface CommandLink extends Command {
    /**
     * A tooltip for the command, when represented in the UI.
     */
    tooltip?: string;
}

interface CommandLinkGroup {
    title?: string;
    commands: CommandLink[];
}
```

Such actions on the client side are appended to a hover bottom as command links:
```
  +-----------------------------+
  | Hover content               |
  |                             |
  +-----------------------------+
  | _Action1_ | _Action2_       |  <- first group, no TITLE
  +-----------------------------+
  | TITLE _Action1_ | _Action2_ |  <- second group
  +-----------------------------+
  ...
```
