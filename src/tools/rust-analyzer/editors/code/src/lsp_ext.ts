/* eslint-disable @typescript-eslint/no-empty-object-type */
/**
 * This file mirrors `crates/rust-analyzer/src/lsp_ext.rs` declarations.
 */

import * as lc from "vscode-languageclient";

// rust-analyzer overrides

export const hover = new lc.RequestType<
    HoverParams,
    (lc.Hover & { actions: CommandLinkGroup[] }) | null,
    void
>(lc.HoverRequest.method);
export type HoverParams = { position: lc.Position | lc.Range } & Omit<lc.HoverParams, "position">;

export type CommandLink = {
    /**
     * A tooltip for the command, when represented in the UI.
     */
    tooltip?: string;
} & lc.Command;
export type CommandLinkGroup = {
    title?: string;
    commands: CommandLink[];
};

// rust-analyzer extensions

export const analyzerStatus = new lc.RequestType<AnalyzerStatusParams, string, void>(
    "rust-analyzer/analyzerStatus",
);
export const cancelFlycheck = new lc.NotificationType0("rust-analyzer/cancelFlycheck");
export const clearFlycheck = new lc.NotificationType0("rust-analyzer/clearFlycheck");
export const expandMacro = new lc.RequestType<ExpandMacroParams, ExpandedMacro | null, void>(
    "rust-analyzer/expandMacro",
);
export const memoryUsage = new lc.RequestType0<string, void>("rust-analyzer/memoryUsage");
export const openServerLogs = new lc.NotificationType0("rust-analyzer/openServerLogs");
export const relatedTests = new lc.RequestType<lc.TextDocumentPositionParams, TestInfo[], void>(
    "rust-analyzer/relatedTests",
);
export const reloadWorkspace = new lc.RequestType0<null, void>("rust-analyzer/reloadWorkspace");
export const rebuildProcMacros = new lc.RequestType0<null, void>("rust-analyzer/rebuildProcMacros");

export const runFlycheck = new lc.NotificationType<{
    textDocument: lc.TextDocumentIdentifier | null;
}>("rust-analyzer/runFlycheck");
export const syntaxTree = new lc.RequestType<SyntaxTreeParams, string, void>(
    "rust-analyzer/syntaxTree",
);
export const viewSyntaxTree = new lc.RequestType<ViewSyntaxTreeParams, string, void>(
    "rust-analyzer/viewSyntaxTree",
);
export const viewCrateGraph = new lc.RequestType<ViewCrateGraphParams, string, void>(
    "rust-analyzer/viewCrateGraph",
);
export const viewFileText = new lc.RequestType<lc.TextDocumentIdentifier, string, void>(
    "rust-analyzer/viewFileText",
);
export const viewHir = new lc.RequestType<lc.TextDocumentPositionParams, string, void>(
    "rust-analyzer/viewHir",
);
export const viewMir = new lc.RequestType<lc.TextDocumentPositionParams, string, void>(
    "rust-analyzer/viewMir",
);
export const interpretFunction = new lc.RequestType<lc.TextDocumentPositionParams, string, void>(
    "rust-analyzer/interpretFunction",
);
export const viewItemTree = new lc.RequestType<ViewItemTreeParams, string, void>(
    "rust-analyzer/viewItemTree",
);

export type DiscoverTestParams = { testId?: string | undefined };
export type RunTestParams = {
    include?: string[] | undefined;
    exclude?: string[] | undefined;
};
export type TestItem = {
    id: string;
    label: string;
    kind: "package" | "module" | "test";
    canResolveChildren: boolean;
    parent?: string | undefined;
    textDocument?: lc.TextDocumentIdentifier | undefined;
    range?: lc.Range | undefined;
    runnable?: Runnable | undefined;
};
export type DiscoverTestResults = {
    tests: TestItem[];
    scope: string[] | undefined;
    scopeFile: lc.TextDocumentIdentifier[] | undefined;
};
export type TestState =
    | { tag: "failed"; message: string }
    | { tag: "passed" }
    | { tag: "started" }
    | { tag: "enqueued" }
    | { tag: "skipped" };
export type ChangeTestStateParams = { testId: string; state: TestState };
export const discoverTest = new lc.RequestType<DiscoverTestParams, DiscoverTestResults, void>(
    "experimental/discoverTest",
);
export const discoveredTests = new lc.NotificationType<DiscoverTestResults>(
    "experimental/discoveredTests",
);
export const runTest = new lc.RequestType<RunTestParams, void, void>("experimental/runTest");
export const abortRunTest = new lc.NotificationType0("experimental/abortRunTest");
export const endRunTest = new lc.NotificationType0("experimental/endRunTest");
export const appendOutputToRunTest = new lc.NotificationType<string>(
    "experimental/appendOutputToRunTest",
);
export const changeTestState = new lc.NotificationType<ChangeTestStateParams>(
    "experimental/changeTestState",
);

export type AnalyzerStatusParams = { textDocument?: lc.TextDocumentIdentifier };

export interface FetchDependencyListParams {}

export interface FetchDependencyListResult {
    crates: {
        name?: string;
        version?: string;
        path: string;
    }[];
}

export const fetchDependencyList = new lc.RequestType<
    FetchDependencyListParams,
    FetchDependencyListResult,
    void
>("rust-analyzer/fetchDependencyList");

export interface FetchDependencyGraphParams {}

export interface FetchDependencyGraphResult {
    crates: {
        name: string;
        version: string;
        path: string;
    }[];
}

export const fetchDependencyGraph = new lc.RequestType<
    FetchDependencyGraphParams,
    FetchDependencyGraphResult,
    void
>("rust-analyzer/fetchDependencyGraph");

export type ExpandMacroParams = {
    textDocument: lc.TextDocumentIdentifier;
    position: lc.Position;
};
export type ExpandedMacro = {
    name: string;
    expansion: string;
};
export type TestInfo = { runnable: Runnable };
export type SyntaxTreeParams = {
    textDocument: lc.TextDocumentIdentifier;
    range: lc.Range | null;
};
export type ViewSyntaxTreeParams = { textDocument: lc.TextDocumentIdentifier };
export type ViewCrateGraphParams = { full: boolean };
export type ViewItemTreeParams = { textDocument: lc.TextDocumentIdentifier };

// experimental extensions

export const joinLines = new lc.RequestType<JoinLinesParams, lc.TextEdit[], void>(
    "experimental/joinLines",
);
export const matchingBrace = new lc.RequestType<MatchingBraceParams, lc.Position[], void>(
    "experimental/matchingBrace",
);
export const moveItem = new lc.RequestType<MoveItemParams, lc.TextEdit[], void>(
    "experimental/moveItem",
);
export const onEnter = new lc.RequestType<lc.TextDocumentPositionParams, lc.TextEdit[], void>(
    "experimental/onEnter",
);
export const openCargoToml = new lc.RequestType<OpenCargoTomlParams, lc.Location, void>(
    "experimental/openCargoToml",
);
export interface DocsUrls {
    local?: string;
    web?: string;
}
export const openDocs = new lc.RequestType<lc.TextDocumentPositionParams, DocsUrls, void>(
    "experimental/externalDocs",
);
export const parentModule = new lc.RequestType<
    lc.TextDocumentPositionParams,
    lc.LocationLink[] | null,
    void
>("experimental/parentModule");
export const childModules = new lc.RequestType<
    lc.TextDocumentPositionParams,
    lc.LocationLink[] | null,
    void
>("experimental/childModules");
export const runnables = new lc.RequestType<RunnablesParams, Runnable[], void>(
    "experimental/runnables",
);
export const serverStatus = new lc.NotificationType<ServerStatusParams>(
    "experimental/serverStatus",
);
export const ssr = new lc.RequestType<SsrParams, lc.WorkspaceEdit, void>("experimental/ssr");
export const viewRecursiveMemoryLayout = new lc.RequestType<
    lc.TextDocumentPositionParams,
    RecursiveMemoryLayout | null,
    void
>("rust-analyzer/viewRecursiveMemoryLayout");

export type JoinLinesParams = {
    textDocument: lc.TextDocumentIdentifier;
    ranges: lc.Range[];
};
export type MatchingBraceParams = {
    textDocument: lc.TextDocumentIdentifier;
    positions: lc.Position[];
};
export type MoveItemParams = {
    textDocument: lc.TextDocumentIdentifier;
    range: lc.Range;
    direction: Direction;
};
export type Direction = "Up" | "Down";
export type OpenCargoTomlParams = {
    textDocument: lc.TextDocumentIdentifier;
};
export type Runnable = {
    label: string;
    location?: lc.LocationLink;
} & (RunnableCargo | RunnableShell);

type RunnableCargo = {
    kind: "cargo";
    args: CargoRunnableArgs;
};

type RunnableShell = {
    kind: "shell";
    args: ShellRunnableArgs;
};

export type CommonRunnableArgs = {
    /**
     * Environment variables to set before running the command.
     */
    environment?: Record<string, string>;
    /**
     * The working directory to run the command in.
     */
    cwd: string;
};

export type ShellRunnableArgs = {
    kind: string;
    program: string;
    args: string[];
} & CommonRunnableArgs;

export type CargoRunnableArgs = {
    /**
     * The workspace root directory of the cargo project.
     */
    workspaceRoot?: string;
    /**
     * Arguments to pass to the executable, these will be passed to the command after a `--` argument.
     */
    executableArgs: string[];
    /**
     * Arguments to pass to cargo.
     */
    cargoArgs: string[];
    /**
     * Command to execute instead of `cargo`.
     */
    // This is supplied by the user via config. We could pull this through the client config in the
    // extension directly, but that would prevent us from honoring the rust-analyzer.toml for it.
    overrideCargo?: string;
} & CommonRunnableArgs;

export type RunnablesParams = {
    textDocument: lc.TextDocumentIdentifier;
    position: lc.Position | null;
};
export type ServerStatusParams = {
    health: "ok" | "warning" | "error";
    quiescent: boolean;
    message?: string;
};
export type SsrParams = {
    query: string;
    parseOnly: boolean;
    textDocument: lc.TextDocumentIdentifier;
    position: lc.Position;
    selections: readonly lc.Range[];
};

export type RecursiveMemoryLayoutNode = {
    item_name: string;
    typename: string;
    size: number;
    alignment: number;
    offset: number;
    parent_idx: number;
    children_start: number;
    children_len: number;
};
export type RecursiveMemoryLayout = {
    nodes: RecursiveMemoryLayoutNode[];
};
