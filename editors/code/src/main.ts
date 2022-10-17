import * as vscode from "vscode";
import * as lc from "vscode-languageclient/node";

import * as commands from "./commands";
import { Ctx, Workspace } from "./ctx";
import { log, isRustDocument } from "./util";
import { PersistentState } from "./persistent_state";
import { activateTaskProvider } from "./tasks";
import { setContextValue } from "./util";
import { bootstrap } from "./bootstrap";
import { Config } from "./config";

const RUST_PROJECT_CONTEXT_NAME = "inRustProject";

export interface RustAnalyzerExtensionApi {
    // FIXME: this should be non-optional
    readonly client?: lc.LanguageClient;
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

    const state = new PersistentState(context.globalState);
    const config = new Config(context);

    const serverPath = await bootstrap(context, config, state).catch((err) => {
        let message = "bootstrap error. ";

        message += 'See the logs in "OUTPUT > Rust Analyzer Client" (should open automatically). ';
        message += 'To enable verbose logs use { "rust-analyzer.trace.extension": true }';

        log.error("Bootstrap error", err);
        throw new Error(message);
    });

    const ctx = new Ctx(context, config, serverPath, workspace);
    // VS Code doesn't show a notification when an extension fails to activate
    // so we do it ourselves.
    return await activateServer(ctx).catch((err) => {
        void vscode.window.showErrorMessage(`Cannot activate rust-analyzer: ${err.message}`);
        throw err;
    });
}

async function activateServer(ctx: Ctx): Promise<RustAnalyzerExtensionApi> {
    if (ctx.workspace.kind === "Workspace Folder") {
        ctx.pushExtCleanup(activateTaskProvider(ctx.config));
    }

    await ctx.activate();
    await initCommonContext(ctx);

    if (ctx.config.typingContinueCommentsOnNewline) {
        ctx.pushExtCleanup(configureLanguage());
    }

    vscode.workspace.onDidChangeConfiguration(
        (_) =>
            ctx
                .getClient()
                .then((it) =>
                    it.sendNotification("workspace/didChangeConfiguration", { settings: "" })
                )
                .catch(log.error),
        null,
        ctx.subscriptions
    );

    return ctx.clientFetcher();
}

async function initCommonContext(ctx: Ctx) {
    // Register a "dumb" onEnter command for the case where server fails to
    // start.
    //
    // FIXME: refactor command registration code such that commands are
    // **always** registered, even if the server does not start. Use API like
    // this perhaps?
    //
    // ```TypeScript
    // registerCommand(
    //    factory: (Ctx) => ((Ctx) => any),
    //    fallback: () => any = () => vscode.window.showErrorMessage(
    //        "rust-analyzer is not available"
    //    ),
    // )
    const defaultOnEnter = vscode.commands.registerCommand("rust-analyzer.onEnter", () =>
        vscode.commands.executeCommand("default:type", { text: "\n" })
    );
    ctx.pushExtCleanup(defaultOnEnter);

    await setContextValue(RUST_PROJECT_CONTEXT_NAME, true);

    // Commands which invokes manually via command palette, shortcut, etc.
    ctx.registerCommand("reload", (_) => async () => {
        void vscode.window.showInformationMessage("Reloading rust-analyzer...");
        await ctx.disposeClient();
        await ctx.activate();
    });

    ctx.registerCommand("analyzerStatus", commands.analyzerStatus);
    ctx.registerCommand("memoryUsage", commands.memoryUsage);
    ctx.registerCommand("shuffleCrateGraph", commands.shuffleCrateGraph);
    ctx.registerCommand("reloadWorkspace", commands.reloadWorkspace);
    ctx.registerCommand("matchingBrace", commands.matchingBrace);
    ctx.registerCommand("joinLines", commands.joinLines);
    ctx.registerCommand("parentModule", commands.parentModule);
    ctx.registerCommand("syntaxTree", commands.syntaxTree);
    ctx.registerCommand("viewHir", commands.viewHir);
    ctx.registerCommand("viewFileText", commands.viewFileText);
    ctx.registerCommand("viewItemTree", commands.viewItemTree);
    ctx.registerCommand("viewCrateGraph", commands.viewCrateGraph);
    ctx.registerCommand("viewFullCrateGraph", commands.viewFullCrateGraph);
    ctx.registerCommand("expandMacro", commands.expandMacro);
    ctx.registerCommand("run", commands.run);
    ctx.registerCommand("copyRunCommandLine", commands.copyRunCommandLine);
    ctx.registerCommand("debug", commands.debug);
    ctx.registerCommand("newDebugConfig", commands.newDebugConfig);
    ctx.registerCommand("openDocs", commands.openDocs);
    ctx.registerCommand("openCargoToml", commands.openCargoToml);
    ctx.registerCommand("peekTests", commands.peekTests);
    ctx.registerCommand("moveItemUp", commands.moveItemUp);
    ctx.registerCommand("moveItemDown", commands.moveItemDown);
    ctx.registerCommand("cancelFlycheck", commands.cancelFlycheck);

    ctx.registerCommand("ssr", commands.ssr);
    ctx.registerCommand("serverVersion", commands.serverVersion);

    // Internal commands which are invoked by the server.
    ctx.registerCommand("runSingle", commands.runSingle);
    ctx.registerCommand("debugSingle", commands.debugSingle);
    ctx.registerCommand("showReferences", commands.showReferences);
    ctx.registerCommand("applySnippetWorkspaceEdit", commands.applySnippetWorkspaceEditCommand);
    ctx.registerCommand("resolveCodeAction", commands.resolveCodeAction);
    ctx.registerCommand("applyActionGroup", commands.applyActionGroup);
    ctx.registerCommand("gotoLocation", commands.gotoLocation);

    ctx.registerCommand("linkToCommand", commands.linkToCommand);

    defaultOnEnter.dispose();
    ctx.registerCommand("onEnter", commands.onEnter);
}

/**
 * Sets up additional language configuration that's impossible to do via a
 * separate language-configuration.json file. See [1] for more information.
 *
 * [1]: https://github.com/Microsoft/vscode/issues/11514#issuecomment-244707076
 */
function configureLanguage(): vscode.Disposable {
    const indentAction = vscode.IndentAction.None;
    return vscode.languages.setLanguageConfiguration("rust", {
        onEnterRules: [
            {
                // Doc single-line comment
                // e.g. ///|
                beforeText: /^\s*\/{3}.*$/,
                action: { indentAction, appendText: "/// " },
            },
            {
                // Parent doc single-line comment
                // e.g. //!|
                beforeText: /^\s*\/{2}\!.*$/,
                action: { indentAction, appendText: "//! " },
            },
            {
                // Begins an auto-closed multi-line comment (standard or parent doc)
                // e.g. /** | */ or /*! | */
                beforeText: /^\s*\/\*(\*|\!)(?!\/)([^\*]|\*(?!\/))*$/,
                afterText: /^\s*\*\/$/,
                action: { indentAction: vscode.IndentAction.IndentOutdent, appendText: " * " },
            },
            {
                // Begins a multi-line comment (standard or parent doc)
                // e.g. /** ...| or /*! ...|
                beforeText: /^\s*\/\*(\*|\!)(?!\/)([^\*]|\*(?!\/))*$/,
                action: { indentAction, appendText: " * " },
            },
            {
                // Continues a multi-line comment
                // e.g.  * ...|
                beforeText: /^(\ \ )*\ \*(\ ([^\*]|\*(?!\/))*)?$/,
                action: { indentAction, appendText: "* " },
            },
            {
                // Dedents after closing a multi-line comment
                // e.g.  */|
                beforeText: /^(\ \ )*\ \*\/\s*$/,
                action: { indentAction, removeText: 1 },
            },
        ],
    });
}
