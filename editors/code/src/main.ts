import * as vscode from "vscode";
import * as lc from "vscode-languageclient/node";

import * as commands from "./commands";
import { CommandFactory, Ctx, fetchWorkspace } from "./ctx";
import * as diagnostics from "./diagnostics";
import { activateTaskProvider } from "./tasks";
import { setContextValue } from "./util";

const RUST_PROJECT_CONTEXT_NAME = "inRustProject";

export interface RustAnalyzerExtensionApi {
    readonly client?: lc.LanguageClient;
}

export async function deactivate() {
    await setContextValue(RUST_PROJECT_CONTEXT_NAME, undefined);
}

export async function activate(
    context: vscode.ExtensionContext
): Promise<RustAnalyzerExtensionApi> {
    if (vscode.extensions.getExtension("rust-lang.rust")) {
        vscode.window
            .showWarningMessage(
                `You have both the rust-analyzer (rust-lang.rust-analyzer) and Rust (rust-lang.rust) ` +
                    "plugins enabled. These are known to conflict and cause various functions of " +
                    "both plugins to not work correctly. You should disable one of them.",
                "Got it"
            )
            .then(() => {}, console.error);
    }

    const ctx = new Ctx(context, createCommands(), fetchWorkspace());
    // VS Code doesn't show a notification when an extension fails to activate
    // so we do it ourselves.
    const api = await activateServer(ctx).catch((err) => {
        void vscode.window.showErrorMessage(
            `Cannot activate rust-analyzer extension: ${err.message}`
        );
        throw err;
    });
    await setContextValue(RUST_PROJECT_CONTEXT_NAME, true);
    return api;
}

async function activateServer(ctx: Ctx): Promise<RustAnalyzerExtensionApi> {
    if (ctx.workspace.kind === "Workspace Folder") {
        ctx.pushExtCleanup(activateTaskProvider(ctx.config));
    }

    const diagnosticProvider = new diagnostics.TextDocumentProvider(ctx);
    ctx.pushExtCleanup(
        vscode.workspace.registerTextDocumentContentProvider(
            diagnostics.URI_SCHEME,
            diagnosticProvider
        )
    );

    const decorationProvider = new diagnostics.AnsiDecorationProvider(ctx);
    ctx.pushExtCleanup(decorationProvider);

    async function decorateVisibleEditors(document: vscode.TextDocument) {
        for (const editor of vscode.window.visibleTextEditors) {
            if (document === editor.document) {
                await decorationProvider.provideDecorations(editor);
            }
        }
    }

    vscode.workspace.onDidChangeTextDocument(
        async (event) => await decorateVisibleEditors(event.document),
        null,
        ctx.subscriptions
    );
    vscode.workspace.onDidOpenTextDocument(decorateVisibleEditors, null, ctx.subscriptions);
    vscode.window.onDidChangeActiveTextEditor(
        async (editor) => {
            if (editor) {
                diagnosticProvider.triggerUpdate(editor.document.uri);
                await decorateVisibleEditors(editor.document);
            }
        },
        null,
        ctx.subscriptions
    );
    vscode.window.onDidChangeVisibleTextEditors(
        async (visibleEditors) => {
            for (const editor of visibleEditors) {
                diagnosticProvider.triggerUpdate(editor.document.uri);
                await decorationProvider.provideDecorations(editor);
            }
        },
        null,
        ctx.subscriptions
    );

    vscode.workspace.onDidChangeWorkspaceFolders(
        async (_) => ctx.onWorkspaceFolderChanges(),
        null,
        ctx.subscriptions
    );
    vscode.workspace.onDidChangeConfiguration(
        async (_) => {
            await ctx.client?.sendNotification(lc.DidChangeConfigurationNotification.type, {
                settings: "",
            });
        },
        null,
        ctx.subscriptions
    );

    await ctx.start();
    return ctx;
}

function createCommands(): Record<string, CommandFactory> {
    return {
        onEnter: {
            enabled: commands.onEnter,
            disabled: (_) => () => vscode.commands.executeCommand("default:type", { text: "\n" }),
        },
        reload: {
            enabled: (ctx) => async () => {
                void vscode.window.showInformationMessage("Reloading rust-analyzer...");
                await ctx.restart();
            },
            disabled: (ctx) => async () => {
                void vscode.window.showInformationMessage("Reloading rust-analyzer...");
                await ctx.start();
            },
        },
        startServer: {
            enabled: (ctx) => async () => {
                await ctx.start();
            },
            disabled: (ctx) => async () => {
                await ctx.start();
            },
        },
        stopServer: {
            enabled: (ctx) => async () => {
                // FIXME: We should re-use the client, that is ctx.deactivate() if none of the configs have changed
                await ctx.stopAndDispose();
                ctx.setServerStatus({
                    health: "stopped",
                });
            },
            disabled: (_) => async () => {},
        },

        analyzerStatus: { enabled: commands.analyzerStatus },
        memoryUsage: { enabled: commands.memoryUsage },
        shuffleCrateGraph: { enabled: commands.shuffleCrateGraph },
        reloadWorkspace: { enabled: commands.reloadWorkspace },
        matchingBrace: { enabled: commands.matchingBrace },
        joinLines: { enabled: commands.joinLines },
        parentModule: { enabled: commands.parentModule },
        syntaxTree: { enabled: commands.syntaxTree },
        viewHir: { enabled: commands.viewHir },
        viewMir: { enabled: commands.viewMir },
        viewFileText: { enabled: commands.viewFileText },
        viewItemTree: { enabled: commands.viewItemTree },
        viewCrateGraph: { enabled: commands.viewCrateGraph },
        viewFullCrateGraph: { enabled: commands.viewFullCrateGraph },
        expandMacro: { enabled: commands.expandMacro },
        run: { enabled: commands.run },
        copyRunCommandLine: { enabled: commands.copyRunCommandLine },
        debug: { enabled: commands.debug },
        newDebugConfig: { enabled: commands.newDebugConfig },
        openDocs: { enabled: commands.openDocs },
        openCargoToml: { enabled: commands.openCargoToml },
        peekTests: { enabled: commands.peekTests },
        moveItemUp: { enabled: commands.moveItemUp },
        moveItemDown: { enabled: commands.moveItemDown },
        cancelFlycheck: { enabled: commands.cancelFlycheck },
        clearFlycheck: { enabled: commands.clearFlycheck },
        runFlycheck: { enabled: commands.runFlycheck },
        ssr: { enabled: commands.ssr },
        serverVersion: { enabled: commands.serverVersion },
        // Internal commands which are invoked by the server.
        applyActionGroup: { enabled: commands.applyActionGroup },
        applySnippetWorkspaceEdit: { enabled: commands.applySnippetWorkspaceEditCommand },
        debugSingle: { enabled: commands.debugSingle },
        gotoLocation: { enabled: commands.gotoLocation },
        linkToCommand: { enabled: commands.linkToCommand },
        resolveCodeAction: { enabled: commands.resolveCodeAction },
        runSingle: { enabled: commands.runSingle },
        showReferences: { enabled: commands.showReferences },
        triggerParameterHints: { enabled: commands.triggerParameterHints },
        openLogs: { enabled: commands.openLogs },
    };
}
