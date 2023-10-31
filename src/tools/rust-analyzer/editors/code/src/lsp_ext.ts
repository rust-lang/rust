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
export const shuffleCrateGraph = new lc.RequestType0<null, void>("rust-analyzer/shuffleCrateGraph");
export const syntaxTree = new lc.RequestType<SyntaxTreeParams, string, void>(
    "rust-analyzer/syntaxTree",
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

export type AnalyzerStatusParams = { textDocument?: lc.TextDocumentIdentifier };

export interface FetchDependencyListParams {}

export interface FetchDependencyListResult {
    crates: {
        name: string | undefined;
        version: string | undefined;
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
export const openDocs = new lc.RequestType<lc.TextDocumentPositionParams, string | void, void>(
    "experimental/externalDocs",
);
export const parentModule = new lc.RequestType<
    lc.TextDocumentPositionParams,
    lc.LocationLink[] | null,
    void
>("experimental/parentModule");
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
    kind: "cargo";
    args: {
        workspaceRoot?: string;
        cargoArgs: string[];
        cargoExtraArgs: string[];
        executableArgs: string[];
        expectTest?: boolean;
        overrideCargo?: string;
    };
};
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
