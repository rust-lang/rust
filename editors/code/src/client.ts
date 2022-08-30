import * as lc from "vscode-languageclient/node";
import * as vscode from "vscode";
import * as ra from "../src/lsp_ext";
import * as Is from "vscode-languageclient/lib/common/utils/is";
import { assert } from "./util";
import { WorkspaceEdit } from "vscode";
import { Workspace } from "./ctx";
import { substituteVariablesInEnv } from "./config";
import { outputChannel, traceOutputChannel } from "./main";
import { randomUUID } from "crypto";

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
setInterval(function cleanupOlderCommandLinks() {
    // keys are returned in insertion order, we'll keep a few
    // of recent keys available, and clean the rest
    const keys = [...LINKED_COMMANDS.keys()];
    const keysToRemove = keys.slice(0, keys.length - 10);
    for (const key of keysToRemove) {
        LINKED_COMMANDS.delete(key);
    }
}, 10 * 60 * 1000);

function renderCommand(cmd: ra.CommandLink): string {
    const commandId = randomUUID();
    LINKED_COMMANDS.set(commandId, cmd);
    return `[${cmd.title}](command:rust-analyzer.linkToCommand?${encodeURIComponent(
        JSON.stringify([commandId])
    )} '${cmd.tooltip}')`;
}

function renderHoverActions(actions: ra.CommandLinkGroup[]): vscode.MarkdownString {
    const text = actions
        .map(
            (group) =>
                (group.title ? group.title + " " : "") +
                group.commands.map(renderCommand).join(" | ")
        )
        .join("___");

    const result = new vscode.MarkdownString(text);
    result.isTrusted = true;
    return result;
}

export async function createClient(
    serverPath: string,
    workspace: Workspace,
    extraEnv: Env
): Promise<lc.LanguageClient> {
    // '.' Is the fallback if no folder is open
    // TODO?: Workspace folders support Uri's (eg: file://test.txt).
    // It might be a good idea to test if the uri points to a file.

    const newEnv = substituteVariablesInEnv(Object.assign({}, process.env, extraEnv));
    const run: lc.Executable = {
        command: serverPath,
        options: { env: newEnv },
    };
    const serverOptions: lc.ServerOptions = {
        run,
        debug: run,
    };

    let initializationOptions = vscode.workspace.getConfiguration("rust-analyzer");

    if (workspace.kind === "Detached Files") {
        initializationOptions = {
            detachedFiles: workspace.files.map((file) => file.uri.fsPath),
            ...initializationOptions,
        };
    }

    const clientOptions: lc.LanguageClientOptions = {
        documentSelector: [{ scheme: "file", language: "rust" }],
        initializationOptions,
        diagnosticCollectionName: "rustc",
        traceOutputChannel: traceOutputChannel(),
        outputChannel: outputChannel(),
        middleware: {
            async provideHover(
                document: vscode.TextDocument,
                position: vscode.Position,
                token: vscode.CancellationToken,
                _next: lc.ProvideHoverSignature
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
                        token
                    )
                    .then(
                        (result) => {
                            const hover = client.protocol2CodeConverter.asHover(result);
                            if (hover) {
                                const actions = (<any>result).actions;
                                if (actions) {
                                    hover.contents.push(renderHoverActions(actions));
                                }
                            }
                            return hover;
                        },
                        (error) => {
                            client.handleFailedRequest(lc.HoverRequest.type, token, error, null);
                            return Promise.resolve(null);
                        }
                    );
            },
            // Using custom handling of CodeActions to support action groups and snippet edits.
            // Note that this means we have to re-implement lazy edit resolving ourselves as well.
            async provideCodeActions(
                document: vscode.TextDocument,
                range: vscode.Range,
                context: vscode.CodeActionContext,
                token: vscode.CancellationToken,
                _next: lc.ProvideCodeActionsSignature
            ) {
                const params: lc.CodeActionParams = {
                    textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(document),
                    range: client.code2ProtocolConverter.asRange(range),
                    context: await client.code2ProtocolConverter.asCodeActionContext(
                        context,
                        token
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
                                    "We don't expect to receive commands in CodeActions"
                                );
                                const action = await client.protocol2CodeConverter.asCodeAction(
                                    item,
                                    token
                                );
                                result.push(action);
                                continue;
                            }
                            assert(
                                isCodeActionWithoutEditsAndCommands(item),
                                "We don't expect edits or commands here"
                            );
                            const kind = client.protocol2CodeConverter.asCodeActionKind(
                                (item as any).kind
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
                                result[index] = items[0];
                            } else {
                                const action = new vscode.CodeAction(group);
                                action.kind = items[0].kind;
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
                    (_error) => undefined
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
        clientOptions
    );

    // To turn on all proposed features use: client.registerProposedFeatures();
    client.registerFeature(new ExperimentalFeatures());

    return client;
}

class ExperimentalFeatures implements lc.StaticFeature {
    fillClientCapabilities(capabilities: lc.ClientCapabilities): void {
        const caps: any = capabilities.experimental ?? {};
        caps.snippetTextEdit = true;
        caps.codeActionGroup = true;
        caps.hoverActions = true;
        caps.serverStatusNotification = true;
        caps.commands = {
            commands: [
                "rust-analyzer.runSingle",
                "rust-analyzer.debugSingle",
                "rust-analyzer.showReferences",
                "rust-analyzer.gotoLocation",
                "editor.action.triggerParameterHints",
            ],
        };
        capabilities.experimental = caps;
    }
    initialize(
        _capabilities: lc.ServerCapabilities<any>,
        _documentSelector: lc.DocumentSelector | undefined
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
