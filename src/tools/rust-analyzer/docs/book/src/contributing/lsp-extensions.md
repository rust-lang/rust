<!---
lsp/ext.rs hash: 78e87a78de8f288e

If you need to change the above hash to make the test pass, please check if you
need to adjust this doc as well and ping this issue:

  https://github.com/rust-lang/rust-analyzer/issues/4604

--->

# LSP Extensions

This document describes LSP extensions used by rust-analyzer.
It's a best effort document, when in doubt, consult the source (and send a PR with clarification ;-) ).
We aim to upstream all non Rust-specific extensions to the protocol, but this is not a top priority.
All capabilities are enabled via the `experimental` field of `ClientCapabilities` or `ServerCapabilities`.
Requests which we hope to upstream live under `experimental/` namespace.
Requests, which are likely to always remain specific to `rust-analyzer` are under `rust-analyzer/` namespace.

If you want to be notified about the changes to this document, subscribe to [#4604](https://github.com/rust-lang/rust-analyzer/issues/4604).

<!-- toc -->

## Configuration in `initializationOptions`

**Upstream Issue:** https://github.com/microsoft/language-server-protocol/issues/567

The `initializationOptions` field of the `InitializeParams` of the initialization request should contain the `"rust-analyzer"` section of the configuration.

`rust-analyzer` normally sends a `"workspace/configuration"` request with `{ "items": ["rust-analyzer"] }` payload.
However, the server can't do this during initialization.
At the same time some essential configuration parameters are needed early on, before servicing requests.
For this reason, we ask that `initializationOptions` contains the configuration, as if the server did make a `"workspace/configuration"` request.

If a language client does not know about `rust-analyzer`'s configuration options it can get sensible defaults by doing any of the following:
 * Not sending `initializationOptions`
 * Sending `"initializationOptions": null`
 * Sending `"initializationOptions": {}`

## Snippet `TextEdit`

**Upstream Issue:** https://github.com/microsoft/language-server-protocol/issues/724

**Experimental Client Capability:** `{ "snippetTextEdit": boolean }`

If this capability is set, `WorkspaceEdit`s returned from `codeAction` requests and `TextEdit`s returned from `textDocument/onTypeFormatting` requests might contain `SnippetTextEdit`s instead of usual `TextEdit`s:

```typescript
interface SnippetTextEdit extends TextEdit {
    insertTextFormat?: InsertTextFormat;
    annotationId?: ChangeAnnotationIdentifier;
}
```

```typescript
export interface TextDocumentEdit {
    textDocument: OptionalVersionedTextDocumentIdentifier;
    edits: (TextEdit | SnippetTextEdit)[];
}
```

When applying such code action or text edit, the editor should insert snippet, with tab stops and placeholders.
At the moment, rust-analyzer guarantees that only a single `TextDocumentEdit` will have edits which can be `InsertTextFormat.Snippet`.
Any additional `TextDocumentEdit`s will only have edits which are `InsertTextFormat.PlainText`.

### Example

"Add `derive`" code action transforms `struct S;` into `#[derive($0)] struct S;`

### Unresolved Questions

* Where exactly are `SnippetTextEdit`s allowed (only in code actions at the moment)?
* Can snippets span multiple files (so far, no)?

## `CodeAction` Groups

**Upstream Issue:** https://github.com/microsoft/language-server-protocol/issues/994

**Experimental Client Capability:** `{ "codeActionGroup": boolean }`

If this capability is set, `CodeAction`s returned from the server contain an additional field, `group`:

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

## Parent Module

**Upstream Issue:** https://github.com/microsoft/language-server-protocol/issues/1002

**Experimental Server Capability:** `{ "parentModule": boolean }`

This request is sent from client to server to handle "Goto Parent Module" editor action.

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
  This is the approach IntelliJ Rust is taking.
  However, experience shows that super module (which generally has a feeling of navigation between files) should be separate.
  If you want super module, but the cursor happens to be inside an overridden function, the behavior with single "gotoSuper" request is surprising.

## Join Lines

**Upstream Issue:** https://github.com/microsoft/language-server-protocol/issues/992

**Experimental Server Capability:** `{ "joinLines": boolean }`

This request is sent from client to server to handle "Join Lines" editor action.

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
  Currently, this is left to editor's discretion, but it might be useful to specify on the server via snippets.
  However, it then becomes unclear how it works with multi cursor.

## On Enter

**Upstream Issue:** https://github.com/microsoft/language-server-protocol/issues/1001

**Experimental Server Capability:** `{ "onEnter": boolean }`

This request is sent from client to server to handle the <kbd>Enter</kbd> key press.

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

As proper cursor positioning is raison d'être for `onEnter`, it uses `SnippetTextEdit`.

### Unresolved Question

* How to deal with synchronicity of the request?
  One option is to require the client to block until the server returns the response.
  Another option is to do a operational transforms style merging of edits from client and server.
  A third option is to do a record-replay: client applies heuristic on enter immediately, then applies all user's keypresses.
  When the server is ready with the response, the client rollbacks all the changes and applies the recorded actions on top of the correct response.
* How to deal with multiple carets?
* Should we extend this to arbitrary typed events and not just `onEnter`?

## Structural Search Replace (SSR)

**Experimental Server Capability:** `{ "ssr": boolean }`

This request is sent from client to server to handle structural search replace -- automated syntax tree based transformation of the source.

**Method:** `experimental/ssr`

**Request:**

```typescript
interface SsrParams {
    /// Search query.
    /// The specific syntax is specified outside of the protocol.
    query: string,
    /// If true, only check the syntax of the query and don't compute the actual edit.
    parseOnly: boolean,
    /// The current text document. This and `position` will be used to determine in what scope
    /// paths in `query` should be resolved.
    textDocument: TextDocumentIdentifier;
    /// Position where SSR was invoked.
    position: Position;
    /// Current selections. Search/replace will be restricted to these if non-empty.
    selections: Range[];
}
```

**Response:**

```typescript
WorkspaceEdit
```

### Example

SSR with query `foo($a, $b) ==>> ($a).foo($b)` will transform, eg `foo(y + 5, z)` into `(y + 5).foo(z)`.

### Unresolved Question

* Probably needs search without replace mode
* Needs a way to limit the scope to certain files.

## Matching Brace

**Upstream Issue:** https://github.com/microsoft/language-server-protocol/issues/999

**Experimental Server Capability:** `{ "matchingBrace": boolean }`

This request is sent from client to server to handle "Matching Brace" editor action.

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
    let x: Vec<()>/*cursor here*/ = vec![];
}
```

`experimental/matchingBrace` yields the position of `<`.
In many cases, matching braces can be handled by the editor.
However, some cases (like disambiguating between generics and comparison operations) need a real parser.
Moreover, it would be cool if editors didn't need to implement even basic language parsing

### Unresolved Question

* Should we return a nested brace structure, to allow [paredit](https://paredit.org/)-like actions of jump *out* of the current brace pair?
  This is how `SelectionRange` request works.
* Alternatively, should we perhaps flag certain `SelectionRange`s as being brace pairs?

## Runnables

**Upstream Issue:** https://github.com/microsoft/language-server-protocol/issues/944

**Experimental Server Capability:** `{ "runnables": { "kinds": string[] } }`

This request is sent from client to server to get the list of things that can be run (tests, binaries, `cargo check -p`).

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
    /// If this Runnable is associated with a specific function/module, etc., the location of this item
    location?: LocationLink;
    /// Running things is necessary technology specific, `kind` needs to be advertised via server capabilities,
    // the type of `args` is specific to `kind`. The actual running is handled by the client.
    kind: string;
    args: any;
}
```

rust-analyzer supports two `kind`s of runnables, `"cargo"` and `"shell"`. The `args` for `"cargo"` look like this:

```typescript
{
    /**
     * Environment variables to set before running the command.
     */
    environment?: Record<string, string>;
    /**
     * The working directory to run the command in.
     */
    cwd: string;
    /**
     * The workspace root directory of the cargo project.
     */
    workspaceRoot?: string;
    /**
     * The cargo command to run.
     */
    cargoArgs: string[];
    /**
     * Arguments to pass to the executable, these will be passed to the command after a `--` argument.
     */
    executableArgs: string[];
    /**
     * Command to execute instead of `cargo`.
     */
    overrideCargo?: string;
}
```

The args for `"shell"` look like this:

```typescript
{
    /**
     * Environment variables to set before running the command.
     */
    environment?: Record<string, string>;
    /**
     * The working directory to run the command in.
     */
    cwd: string;
    kind: string;
    program: string;
    args: string[];
}
```

## Test explorer

**Experimental Client Capability:** `{ "testExplorer": boolean }`

If this capability is set, the `experimental/discoveredTests` notification will be sent from the
server to the client.

**Method:** `experimental/discoverTest`

**Request:** `DiscoverTestParams`

```typescript
interface DiscoverTestParams {
    // The test that we need to resolve its children. If not present,
    // the response should return top level tests.
    testId?: string | undefined;
}
```

**Response:** `DiscoverTestResults`

```typescript
interface TestItem {
    // A unique identifier for the test
    id: string;
    // The file containing this test
    textDocument?: lc.TextDocumentIdentifier | undefined;
    // The range in the file containing this test
    range?: lc.Range | undefined;
    // A human readable name for this test
    label: string;
    // The kind of this test item. Based on the kind,
	// an icon is chosen by the editor.
    kind: "package" | "module" | "test";
    // True if this test may have children not available eagerly
    canResolveChildren: boolean;
    // The id of the parent test in the test tree. If not present, this test
    // is a top level test.
    parent?: string | undefined;
    // The information useful for running the test. The client can use `runTest`
    // request for simple execution, but for more complex execution forms
    // like debugging, this field is useful.
    // Note that this field includes some information about label and location as well, but
    // those exist just for keeping things in sync with other methods of running runnables
    // (for example using one consistent name in the vscode's launch.json) so for any propose
    // other than running tests this field should not be used.
    runnable?: Runnable | undefined;
};

interface DiscoverTestResults {
    // The discovered tests.
    tests: TestItem[];
    // For each test which its id is in this list, the response
    // contains all tests that are children of this test, and
    // client should remove old tests not included in the response.
    scope: string[] | undefined;
    // For each file which its uri is in this list, the response
    // contains all tests that are located in this file, and
    // client should remove old tests not included in the response.
    scopeFile: lc.TextDocumentIdentifier[] | undefined;
}
```

**Method:** `experimental/discoveredTests`

**Notification:** `DiscoverTestResults`

This notification is sent from the server to the client when the
server detect changes in the existing tests. The `DiscoverTestResults` is
the same as the one in `experimental/discoverTest` response.

**Method:** `experimental/runTest`

**Request:** `RunTestParams`

```typescript
interface RunTestParams {
    // Id of the tests to be run. If a test is included, all of its children are included implicitly. If
    // this property is undefined, then the server should simply run all tests.
    include?: string[] | undefined;
    // An array of test ids the user has marked as excluded from the test included in this run; exclusions
    // should apply after inclusions.
    // May be omitted if no exclusions were requested. Server should not run excluded tests or
    // any children of excluded tests.
    exclude?: string[] | undefined;
}
```

**Response:** `void`

**Method:** `experimental/endRunTest`

**Notification:**

This notification is sent from the server to the client when the current running
session is finished. The server should not send any run notification
after this.

**Method:** `experimental/abortRunTest`

**Notification:**

This notification is sent from the client to the server when the user is no longer
interested in the test results. The server should clean up its resources and send
a `experimental/endRunTest` when is done.

**Method:** `experimental/changeTestState`

**Notification:** `ChangeTestStateParams`

```typescript
type TestState = { tag: "passed" }
    | {
        tag: "failed";
        // The standard error of the test, containing the panic message. Clients should
        // render it similar to a terminal, and e.g. handle ansi colors.
        message: string;
    }
    | { tag: "started" }
    | { tag: "enqueued" }
    | { tag: "skipped" };

interface ChangeTestStateParams {
    testId: string;
    state: TestState;
}
```

**Method:** `experimental/appendOutputToRunTest`

**Notification:** `string`

This notification is used for reporting messages independent of any single test and related to the run session
in general, e.g. cargo compiling progress messages or warnings.

## Open External Documentation

This request is sent from the client to the server to obtain web and local URL(s) for documentation related to the symbol under the cursor, if available.

**Method:** `experimental/externalDocs`

**Request:** `TextDocumentPositionParams`

**Response:** `string | null`

## Local Documentation

**Experimental Client Capability:** `{ "localDocs": boolean }`

If this capability is set, the `Open External Documentation` request returned from the server will have the following structure:

```typescript
interface ExternalDocsResponse {
    web?: string;
    local?: string;
}
```

## Analyzer Status

**Method:** `rust-analyzer/analyzerStatus`

**Request:**

```typescript
interface AnalyzerStatusParams {
    /// If specified, show dependencies of the current file.
    textDocument?: TextDocumentIdentifier;
}
```

**Response:** `string`

Returns internal status message, mostly for debugging purposes.

## Reload Workspace

**Method:** `rust-analyzer/reloadWorkspace`

**Request:** `null`

**Response:** `null`

Reloads project information (that is, re-executes `cargo metadata`).

## Rebuild proc-macros

**Method:** `rust-analyzer/rebuildProcMacros`

**Request:** `null`

**Response:** `null`

Rebuilds build scripts and proc-macros, and runs the build scripts to reseed the build data.

## Server Status

**Experimental Client Capability:** `{ "serverStatusNotification": boolean }`

**Method:** `experimental/serverStatus`

**Notification:**

```typescript
interface ServerStatusParams {
    /// `ok` means that the server is completely functional.
    ///
    /// `warning` means that the server is partially functional.
    /// It can answer correctly to most requests, but some results
    /// might be wrong due to, for example, some missing dependencies.
    ///
    /// `error` means that the server is not functional. For example,
    /// there's a fatal build configuration problem. The server might
    /// still give correct answers to simple requests, but most results
    /// will be incomplete or wrong.
    health: "ok" | "warning" | "error",
    /// Is there any pending background work which might change the status?
    /// For example, are dependencies being downloaded?
    quiescent: boolean,
    /// Explanatory message to show on hover.
    message?: string,
}
```

This notification is sent from server to client.
The client can use it to display *persistent* status to the user (in modline).
It is similar to the `showMessage`, but is intended for stares rather than point-in-time events.

Note that this functionality is intended primarily to inform the end user about the state of the server.
In particular, it's valid for the client to completely ignore this extension.
Clients are discouraged from but are allowed to use the `health` status to decide if it's worth sending a request to the server.

### Controlling Flycheck

The flycheck/checkOnSave feature can be controlled via notifications sent by the client to the server.

**Method:** `rust-analyzer/runFlycheck`

**Notification:**

```typescript
interface RunFlycheckParams {
    /// The text document whose cargo workspace flycheck process should be started.
    /// If the document is null or does not belong to a cargo workspace all flycheck processes will be started.
    textDocument: lc.TextDocumentIdentifier | null;
}
```

Triggers the flycheck processes.


**Method:** `rust-analyzer/clearFlycheck`

**Notification:**

```typescript
interface ClearFlycheckParams {}
```

Clears the flycheck diagnostics.

**Method:** `rust-analyzer/cancelFlycheck`

**Notification:**

```typescript
interface CancelFlycheckParams {}
```

Cancels all running flycheck processes.

## View Syntax Tree

**Method:** `rust-analyzer/viewSyntaxTree`

**Request:**

```typescript
interface ViewSyntaxTreeParams {
    textDocument: TextDocumentIdentifier,
}
```

**Response:** `string`

Returns json representation of the file's syntax tree.
Used to create a treeView for debugging and working on rust-analyzer itself.

## View Hir

**Method:** `rust-analyzer/viewHir`

**Request:** `TextDocumentPositionParams`

**Response:** `string`

Returns a textual representation of the HIR of the function containing the cursor.
For debugging or when working on rust-analyzer itself.

## View Mir

**Method:** `rust-analyzer/viewMir`

**Request:** `TextDocumentPositionParams`

**Response:** `string`

Returns a textual representation of the MIR of the function containing the cursor.
For debugging or when working on rust-analyzer itself.

## Interpret Function

**Method:** `rust-analyzer/interpretFunction`

**Request:** `TextDocumentPositionParams`

**Response:** `string`

Tries to evaluate the function using internal rust analyzer knowledge, without compiling
the code. Currently evaluates the function under cursor, but will give a runnable in
future. Highly experimental.

## View File Text

**Method:** `rust-analyzer/viewFileText`

**Request:** `TextDocumentIdentifier`

**Response:** `string`

Returns the text of a file as seen by the server.
This is for debugging file sync problems.

## View ItemTree

**Method:** `rust-analyzer/viewItemTree`

**Request:**

```typescript
interface ViewItemTreeParams {
    textDocument: TextDocumentIdentifier,
}
```

**Response:** `string`

Returns a textual representation of the `ItemTree` of the currently open file, for debugging.

## View Crate Graph

**Method:** `rust-analyzer/viewCrateGraph`

**Request:**

```typescript
interface ViewCrateGraphParams {
    full: boolean,
}
```

**Response:** `string`

Renders rust-analyzer's crate graph as an SVG image.

If `full` is `true`, the graph includes non-workspace crates (crates.io dependencies as well as sysroot crates).

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

## Hover Actions

**Experimental Client Capability:** `{ "hoverActions": boolean }`

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

## Open Cargo.toml

**Upstream Issue:** https://github.com/rust-lang/rust-analyzer/issues/6462

**Experimental Server Capability:** `{ "openCargoToml": boolean }`

This request is sent from client to server to open the current project's Cargo.toml

**Method:** `experimental/openCargoToml`

**Request:** `OpenCargoTomlParams`

**Response:** `Location | null`


### Example

```rust
// Cargo.toml
[package]
// src/main.rs

/* cursor here*/
```

`experimental/openCargoToml` returns a single `Link` to the start of the `[package]` keyword.

## Related tests

This request is sent from client to server to get the list of tests for the specified position.

**Method:** `rust-analyzer/relatedTests`

**Request:** `TextDocumentPositionParams`

**Response:** `TestInfo[]`

```typescript
interface TestInfo {
    runnable: Runnable;
}
```

## Hover Range

**Upstream Issue:** https://github.com/microsoft/language-server-protocol/issues/377

**Experimental Server Capability:** { "hoverRange": boolean }

This extension allows passing a `Range` as a `position` field of `HoverParams`.
The primary use-case is to use the hover request to show the type of the expression currently selected.

```typescript
interface HoverParams extends WorkDoneProgressParams {
    textDocument: TextDocumentIdentifier;
    position: Range | Position;
}
```
Whenever the client sends a `Range`, it is understood as the current selection and any hover included in the range will show the type of the expression if possible.

### Example

```rust
fn main() {
    let expression = $01 + 2 * 3$0;
}
```

Triggering a hover inside the selection above will show a result of `i32`.

## Move Item

**Upstream Issue:** https://github.com/rust-lang/rust-analyzer/issues/6823

This request is sent from client to server to move item under cursor or selection in some direction.

**Method:** `experimental/moveItem`

**Request:** `MoveItemParams`

**Response:** `SnippetTextEdit[]`

```typescript
export interface MoveItemParams {
    textDocument: TextDocumentIdentifier,
    range: Range,
    direction: Direction
}

export const enum Direction {
    Up = "Up",
    Down = "Down"
}
```

## Workspace Symbols Filtering

**Upstream Issue:** https://github.com/microsoft/language-server-protocol/issues/941

**Experimental Server Capability:** `{ "workspaceSymbolScopeKindFiltering": boolean }`

Extends the existing `workspace/symbol` request with ability to filter symbols by broad scope and kind of symbol.
If this capability is set, `workspace/symbol` parameter gains two new optional fields:


```typescript
interface WorkspaceSymbolParams {
    /**
     * Return only the symbols defined in the specified scope.
     */
    searchScope?: WorkspaceSymbolSearchScope;
    /**
     * Return only the symbols of specified kinds.
     */
    searchKind?: WorkspaceSymbolSearchKind;
    ...
}

const enum WorkspaceSymbolSearchScope {
    Workspace = "workspace",
    WorkspaceAndDependencies = "workspaceAndDependencies"
}

const enum WorkspaceSymbolSearchKind {
    OnlyTypes = "onlyTypes",
    AllSymbols = "allSymbols"
}
```

## Client Commands

**Upstream Issue:** https://github.com/microsoft/language-server-protocol/issues/642

**Experimental Client Capability:** `{ "commands?": ClientCommandOptions }`

Certain LSP types originating on the server, notably code lenses, embed commands.
Commands can be serviced either by the server or by the client.
However, the server doesn't know which commands are available on the client.

This extensions allows the client to communicate this info.


```typescript
export interface ClientCommandOptions {
    /**
     * The commands to be executed on the client
     */
    commands: string[];
}
```

## Colored Diagnostic Output

**Experimental Client Capability:** `{ "colorDiagnosticOutput": boolean }`

If this capability is set, the "full compiler diagnostics" provided by `checkOnSave`
will include ANSI color and style codes to render the diagnostic in a similar manner
as `cargo`. This is translated into `--message-format=json-diagnostic-rendered-ansi`
when flycheck is run, instead of the default `--message-format=json`.

The full compiler rendered diagnostics are included in the server response
regardless of this capability:

```typescript
// https://microsoft.github.io/language-server-protocol/specifications/specification-current#diagnostic
export interface Diagnostic {
    ...
    data?: {
        /**
         * The human-readable compiler output as it would be printed to a terminal.
         * Includes ANSI color and style codes if the client has set the experimental
         * `colorDiagnosticOutput` capability.
         */
        rendered?: string;
    };
}
```

## Dependency Tree

**Method:** `rust-analyzer/fetchDependencyList`

**Request:**

```typescript
export interface FetchDependencyListParams {}
```

**Response:**
```typescript
export interface FetchDependencyListResult {
    crates: {
        name: string;
        version: string;
        path: string;
    }[];
}
```
Returns all crates from this workspace, so it can be used create a viewTree to help navigate the dependency tree.

## View Recursive Memory Layout

**Method:** `rust-analyzer/viewRecursiveMemoryLayout`

**Request:** `TextDocumentPositionParams`

**Response:**

```typescript
export interface RecursiveMemoryLayoutNode = {
    /// Name of the item, or [ROOT], `.n` for tuples
    item_name: string;
    /// Full name of the type (type aliases are ignored)
    typename: string;
    /// Size of the type in bytes
    size: number;
    /// Alignment of the type in bytes
    alignment: number;
    /// Offset of the type relative to its parent (or 0 if its the root)
    offset: number;
    /// Index of the node's parent (or -1 if its the root)
    parent_idx: number;
    /// Index of the node's children (or -1 if it does not have children)
    children_start: number;
    /// Number of child nodes (unspecified it does not have children)
    children_len: number;
};

export interface RecursiveMemoryLayout = {
    nodes: RecursiveMemoryLayoutNode[];
};
```

Returns a vector of nodes representing items in the datatype as a tree, `RecursiveMemoryLayout::nodes[0]` is the root node.

If `RecursiveMemoryLayout::nodes::length == 0` we could not find a suitable type.

Generic Types do not give anything because they are incomplete. Fully specified generic types do not give anything if they are selected directly but do work when a child of other types [this is consistent with other behavior](https://github.com/rust-lang/rust-analyzer/issues/15010).

### Unresolved questions:

- How should enums/unions be represented? currently they do not produce any children because they have multiple distinct sets of children.
- Should niches be represented? currently they are not reported.
- A visual representation of the memory layout is not specified, see the provided implementation for an example, however it may not translate well to terminal based editors or other such things.
