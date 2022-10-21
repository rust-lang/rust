import * as vscode from "vscode";
import * as lc from "vscode-languageclient/node";

import * as commands from "./commands";
import { CommandFactory, Ctx, Workspace } from "./ctx";
import { isRustDocument } from "./util";
import { activateTaskProvider } from "./tasks";
import { setContextValue } from "./util";

const RUST_PROJECT_CONTEXT_NAME = "inRustProject";

export interface RustAnalyzerExtensionApi {
    // FIXME: this should be non-optional
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

    // We only support local folders, not eg. Live Share (`vlsl:` scheme), so don't activate if
    // only those are in use.
    // (r-a still somewhat works with Live Share, because commands are tunneled to the host)
    const folders = (vscode.workspace.workspaceFolders || []).filter(
        (folder) => folder.uri.scheme === "file"
    );
    const rustDocuments = vscode.workspace.textDocuments.filter((document) =>
        isRustDocument(document)
    );

    if (folders.length === 0 && rustDocuments.length === 0) {
        // FIXME: Ideally we would choose not to activate at all (and avoid registering
        // non-functional editor commands), but VS Code doesn't seem to have a good way of doing
        // that
        return {};
    }

    const workspace: Workspace =
        folders.length === 0
            ? {
                  kind: "Detached Files",
                  files: rustDocuments,
              }
            : { kind: "Workspace Folder" };

    const ctx = new Ctx(context, workspace, createCommands());
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

    vscode.workspace.onDidChangeConfiguration(
        async (_) => {
            await ctx
                .clientFetcher()
                .client?.sendNotification("workspace/didChangeConfiguration", { settings: "" });
        },
        null,
        ctx.subscriptions
    );

    await ctx.activate();
    return ctx.clientFetcher();
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
                // FIXME: We should re-use the client, that is ctx.deactivate() if none of the configs have changed
                await ctx.stop();
                await ctx.activate();
            },
            disabled: (ctx) => async () => {
                void vscode.window.showInformationMessage("Reloading rust-analyzer...");
                await ctx.activate();
            },
        },
        startServer: {
            enabled: (ctx) => async () => {
                await ctx.activate();
            },
            disabled: (ctx) => async () => {
                await ctx.activate();
            },
        },
        stopServer: {
            enabled: (ctx) => async () => {
                // FIXME: We should re-use the client, that is ctx.deactivate() if none of the configs have changed
                await ctx.stop();
                ctx.setServerStatus({
                    health: "ok",
                    quiescent: true,
                    message: "server is not running",
                });
            },
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
    };
}
