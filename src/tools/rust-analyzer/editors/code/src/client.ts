import * as anser from "anser";
import * as lc from "vscode-languageclient/node";
import * as vscode from "vscode";
import * as ra from "../src/lsp_ext";
import * as Is from "vscode-languageclient/lib/common/utils/is";
import { assert } from "./util";
import * as diagnostics from "./diagnostics";
import { WorkspaceEdit } from "vscode";
import { type Config, prepareVSCodeConfig } from "./config";
import { randomUUID } from "crypto";
import { sep as pathSeparator } from "path";
import { unwrapUndefinable } from "./undefinable";

export interface Env {
    [name: string]: string;
}

// Command URIs have a form of command:command-name?arguments, where
// arguments is a percent-encoded array of data we want to pass along to
// the command function. For "Show References" this is a list of all file
// URIs with locations of every reference, and it can get quite long.
//
// To work around it we use an intermediary linkToCommand command. When
// we render a command link, a reference to a command with all its arguments
// is stored in a map, and instead a linkToCommand link is rendered
// with the key to that map.
export const LINKED_COMMANDS = new Map<string, ra.CommandLink>();

// For now the map is cleaned up periodically (I've set it to every
// 10 minutes). In general case we'll probably need to introduce TTLs or
// flags to denote ephemeral links (like these in hover popups) and
// persistent links and clean those separately. But for now simply keeping
// the last few links in the map should be good enough. Likewise, we could
// add code to remove a target command from the map after the link is
// clicked, but assuming most links in hover sheets won't be clicked anyway
// this code won't change the overall memory use much.
setInterval(
    function cleanupOlderCommandLinks() {
        // keys are returned in insertion order, we'll keep a few
        // of recent keys available, and clean the rest
        const keys = [...LINKED_COMMANDS.keys()];
        const keysToRemove = keys.slice(0, keys.length - 10);
        for (const key of keysToRemove) {
            LINKED_COMMANDS.delete(key);
        }
    },
    10 * 60 * 1000,
);

function renderCommand(cmd: ra.CommandLink): string {
    const commandId = randomUUID();
    LINKED_COMMANDS.set(commandId, cmd);
    return `[${cmd.title}](command:rust-analyzer.linkToCommand?${encodeURIComponent(
        JSON.stringify([commandId]),
    )} '${cmd.tooltip}')`;
}

function renderHoverActions(actions: ra.CommandLinkGroup[]): vscode.MarkdownString {
    const text = actions
        .map(
            (group) =>
                (group.title ? group.title + " " : "") +
                group.commands.map(renderCommand).join(" | "),
        )
        .join("___");

    const result = new vscode.MarkdownString(text);
    result.isTrusted = true;
    return result;
}

export async function createClient(
    traceOutputChannel: vscode.OutputChannel,
    outputChannel: vscode.OutputChannel,
    initializationOptions: vscode.WorkspaceConfiguration,
    serverOptions: lc.ServerOptions,
    config: Config,
    unlinkedFiles: vscode.Uri[],
): Promise<lc.LanguageClient> {
    const clientOptions: lc.LanguageClientOptions = {
        documentSelector: [{ scheme: "file", language: "rust" }],
        initializationOptions,
        diagnosticCollectionName: "rustc",
        traceOutputChannel,
        outputChannel,
        middleware: {
            workspace: {
                // HACK: This is a workaround, when the client has been disposed, VSCode
                // continues to emit events to the client and the default one for this event
                // attempt to restart the client for no reason
                async didChangeWatchedFile(event, next) {
                    if (client.isRunning()) {
                        await next(event);
                    }
                },
                async configuration(
                    params: lc.ConfigurationParams,
                    token: vscode.CancellationToken,
                    next: lc.ConfigurationRequest.HandlerSignature,
                ) {
                    const resp = await next(params, token);
                    if (resp && Array.isArray(resp)) {
                        return resp.map((val) => {
                            return prepareVSCodeConfig(val, (key, cfg) => {
                                // we only want to set discovered workspaces on the right key
                                // and if a workspace has been discovered.
                                if (
                                    key === "linkedProjects" &&
                                    config.discoveredWorkspaces.length > 0
                                ) {
                                    cfg[key] = config.discoveredWorkspaces;
                                }
                            });
                        });
                    } else {
                        return resp;
                    }
                },
            },
            async handleDiagnostics(
                uri: vscode.Uri,
                diagnosticList: vscode.Diagnostic[],
                next: lc.HandleDiagnosticsSignature,
            ) {
                const preview = config.previewRustcOutput;
                const errorCode = config.useRustcErrorCode;
                diagnosticList.forEach((diag, idx) => {
                    const value =
                        typeof diag.code === "string" || typeof diag.code === "number"
                            ? diag.code
                            : diag.code?.value;
                    if (
                        value === "unlinked-file" &&
                        !unlinkedFiles.includes(uri) &&
                        diag.message !== "file not included in module tree"
                    ) {
                        const config = vscode.workspace.getConfiguration("rust-analyzer");
                        if (config.get("showUnlinkedFileNotification")) {
                            unlinkedFiles.push(uri);
                            const folder = vscode.workspace.getWorkspaceFolder(uri)?.uri.fsPath;
                            if (folder) {
                                const parentBackslash = uri.fsPath.lastIndexOf(
                                    pathSeparator + "src",
                                );
                                const parent = uri.fsPath.substring(0, parentBackslash);

                                if (parent.startsWith(folder)) {
                                    const path = vscode.Uri.file(
                                        parent + pathSeparator + "Cargo.toml",
                                    );
                                    void vscode.workspace.fs.stat(path).then(async () => {
                                        const choice = await vscode.window.showInformationMessage(
                                            `This rust file does not belong to a loaded cargo project. It looks like it might belong to the workspace at ${path.path}, do you want to add it to the linked Projects?`,
                                            "Yes",
                                            "No",
                                            "Don't show this again",
                                        );
                                        switch (choice) {
                                            case undefined:
                                                break;
                                            case "No":
                                                break;
                                            case "Yes":
                                                const pathToInsert =
                                                    "." +
                                                    parent.substring(folder.length) +
                                                    pathSeparator +
                                                    "Cargo.toml";
                                                await config.update(
                                                    "linkedProjects",
                                                    config
                                                        .get<any[]>("linkedProjects")
                                                        ?.concat(pathToInsert),
                                                    false,
                                                );
                                                break;
                                            case "Don't show this again":
                                                await config.update(
                                                    "showUnlinkedFileNotification",
                                                    false,
                                                    false,
                                                );
                                                break;
                                        }
                                    });
                                }
                            }
                        }
                    }

                    // Abuse the fact that VSCode leaks the LSP diagnostics data field through the
                    // Diagnostic class, if they ever break this we are out of luck and have to go
                    // back to the worst diagnostics experience ever:)

                    // We encode the rendered output of a rustc diagnostic in the rendered field of
                    // the data payload of the lsp diagnostic. If that field exists, overwrite the
                    // diagnostic code such that clicking it opens the diagnostic in a readonly
                    // text editor for easy inspection
                    const rendered = (diag as unknown as { data?: { rendered?: string } }).data
                        ?.rendered;
                    if (rendered) {
                        if (preview) {
                            const decolorized = anser.ansiToText(rendered);
                            const index =
                                decolorized.match(/^(note|help):/m)?.index || rendered.length;
                            diag.message = decolorized
                                .substring(0, index)
                                .replace(/^ -->[^\n]+\n/m, "");
                        }
                        diag.code = {
                            target: vscode.Uri.from({
                                scheme: diagnostics.URI_SCHEME,
                                path: `/diagnostic message [${idx.toString()}]`,
                                fragment: uri.toString(),
                                query: idx.toString(),
                            }),
                            value:
                                errorCode && value ? value : "Click for full compiler diagnostic",
                        };
                    }
                });
                return next(uri, diagnosticList);
            },
            async provideHover(
                document: vscode.TextDocument,
                position: vscode.Position,
                token: vscode.CancellationToken,
                _next: lc.ProvideHoverSignature,
            ) {
                const editor = vscode.window.activeTextEditor;
                const positionOrRange = editor?.selection?.contains(position)
                    ? client.code2ProtocolConverter.asRange(editor.selection)
                    : client.code2ProtocolConverter.asPosition(position);
                return client
                    .sendRequest(
                        ra.hover,
                        {
                            textDocument:
                                client.code2ProtocolConverter.asTextDocumentIdentifier(document),
                            position: positionOrRange,
                        },
                        token,
                    )
                    .then(
                        (result) => {
                            if (!result) return null;
                            const hover = client.protocol2CodeConverter.asHover(result);
                            if (!!result.actions) {
                                hover.contents.push(renderHoverActions(result.actions));
                            }
                            return hover;
                        },
                        (error) => {
                            client.handleFailedRequest(lc.HoverRequest.type, token, error, null);
                            return Promise.resolve(null);
                        },
                    );
            },
            // Using custom handling of CodeActions to support action groups and snippet edits.
            // Note that this means we have to re-implement lazy edit resolving ourselves as well.
            async provideCodeActions(
                document: vscode.TextDocument,
                range: vscode.Range,
                context: vscode.CodeActionContext,
                token: vscode.CancellationToken,
                _next: lc.ProvideCodeActionsSignature,
            ) {
                const params: lc.CodeActionParams = {
                    textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(document),
                    range: client.code2ProtocolConverter.asRange(range),
                    context: await client.code2ProtocolConverter.asCodeActionContext(
                        context,
                        token,
                    ),
                };
                return client.sendRequest(lc.CodeActionRequest.type, params, token).then(
                    async (values) => {
                        if (values === null) return undefined;
                        const result: (vscode.CodeAction | vscode.Command)[] = [];
                        const groups = new Map<
                            string,
                            { index: number; items: vscode.CodeAction[] }
                        >();
                        for (const item of values) {
                            // In our case we expect to get code edits only from diagnostics
                            if (lc.CodeAction.is(item)) {
                                assert(
                                    !item.command,
                                    "We don't expect to receive commands in CodeActions",
                                );
                                const action = await client.protocol2CodeConverter.asCodeAction(
                                    item,
                                    token,
                                );
                                result.push(action);
                                continue;
                            }
                            assert(
                                isCodeActionWithoutEditsAndCommands(item),
                                "We don't expect edits or commands here",
                            );
                            const kind = client.protocol2CodeConverter.asCodeActionKind(
                                (item as any).kind,
                            );
                            const action = new vscode.CodeAction(item.title, kind);
                            const group = (item as any).group;
                            action.command = {
                                command: "rust-analyzer.resolveCodeAction",
                                title: item.title,
                                arguments: [item],
                            };

                            // Set a dummy edit, so that VS Code doesn't try to resolve this.
                            action.edit = new WorkspaceEdit();

                            if (group) {
                                let entry = groups.get(group);
                                if (!entry) {
                                    entry = { index: result.length, items: [] };
                                    groups.set(group, entry);
                                    result.push(action);
                                }
                                entry.items.push(action);
                            } else {
                                result.push(action);
                            }
                        }
                        for (const [group, { index, items }] of groups) {
                            if (items.length === 1) {
                                const item = unwrapUndefinable(items[0]);
                                result[index] = item;
                            } else {
                                const action = new vscode.CodeAction(group);
                                const item = unwrapUndefinable(items[0]);
                                action.kind = item.kind;
                                action.command = {
                                    command: "rust-analyzer.applyActionGroup",
                                    title: "",
                                    arguments: [
                                        items.map((item) => {
                                            return {
                                                label: item.title,
                                                arguments: item.command!.arguments![0],
                                            };
                                        }),
                                    ],
                                };

                                // Set a dummy edit, so that VS Code doesn't try to resolve this.
                                action.edit = new WorkspaceEdit();

                                result[index] = action;
                            }
                        }
                        return result;
                    },
                    (_error) => undefined,
                );
            },
        },
        markdown: {
            supportHtml: true,
        },
    };

    const client = new lc.LanguageClient(
        "rust-analyzer",
        "Rust Analyzer Language Server",
        serverOptions,
        clientOptions,
    );

    // To turn on all proposed features use: client.registerProposedFeatures();
    client.registerFeature(new ExperimentalFeatures());
    client.registerFeature(new OverrideFeatures());

    return client;
}

class ExperimentalFeatures implements lc.StaticFeature {
    getState(): lc.FeatureState {
        return { kind: "static" };
    }
    fillClientCapabilities(capabilities: lc.ClientCapabilities): void {
        capabilities.experimental = {
            snippetTextEdit: true,
            codeActionGroup: true,
            hoverActions: true,
            serverStatusNotification: true,
            colorDiagnosticOutput: true,
            openServerLogs: true,
            commands: {
                commands: [
                    "rust-analyzer.runSingle",
                    "rust-analyzer.debugSingle",
                    "rust-analyzer.showReferences",
                    "rust-analyzer.gotoLocation",
                    "editor.action.triggerParameterHints",
                ],
            },
            ...capabilities.experimental,
        };
    }
    initialize(
        _capabilities: lc.ServerCapabilities,
        _documentSelector: lc.DocumentSelector | undefined,
    ): void {}
    dispose(): void {}
}

class OverrideFeatures implements lc.StaticFeature {
    getState(): lc.FeatureState {
        return { kind: "static" };
    }
    fillClientCapabilities(capabilities: lc.ClientCapabilities): void {
        // Force disable `augmentsSyntaxTokens`, VSCode's textmate grammar is somewhat incomplete
        // making the experience generally worse
        const caps = capabilities.textDocument?.semanticTokens;
        if (caps) {
            caps.augmentsSyntaxTokens = false;
        }
    }
    initialize(
        _capabilities: lc.ServerCapabilities,
        _documentSelector: lc.DocumentSelector | undefined,
    ): void {}
    dispose(): void {}
}

function isCodeActionWithoutEditsAndCommands(value: any): boolean {
    const candidate: lc.CodeAction = value;
    return (
        candidate &&
        Is.string(candidate.title) &&
        (candidate.diagnostics === void 0 ||
            Is.typedArray(candidate.diagnostics, lc.Diagnostic.is)) &&
        (candidate.kind === void 0 || Is.string(candidate.kind)) &&
        candidate.edit === void 0 &&
        candidate.command === void 0
    );
}
