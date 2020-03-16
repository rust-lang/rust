import * as lc from "vscode-languageclient";
import * as vscode from "vscode";
import { promises as dns } from "dns";
import { strict as nativeAssert } from "assert";

export function assert(condition: boolean, explanation: string): asserts condition {
    try {
        nativeAssert(condition, explanation);
    } catch (err) {
        log.error(`Assertion failed:`, explanation);
        throw err;
    }
}

export const log = new class {
    private enabled = true;

    setEnabled(yes: boolean): void {
        log.enabled = yes;
    }

    debug(message?: any, ...optionalParams: any[]): void {
        if (!log.enabled) return;
        // eslint-disable-next-line no-console
        console.log(message, ...optionalParams);
    }

    error(message?: any, ...optionalParams: any[]): void {
        if (!log.enabled) return;
        debugger;
        // eslint-disable-next-line no-console
        console.error(message, ...optionalParams);
    }

    downloadError(err: Error, artifactName: string, repoName: string) {
        vscode.window.showErrorMessage(
            `Failed to download the rust-analyzer ${artifactName} from ${repoName} ` +
            `GitHub repository: ${err.message}`
        );
        log.error(err);
        dns.resolve('example.com').then(
            addrs => log.debug("DNS resolution for example.com was successful", addrs),
            err => log.error(
                "DNS resolution for example.com failed, " +
                "there might be an issue with Internet availability",
                err
            )
        );
    }
};

export async function sendRequestWithRetry<TParam, TRet>(
    client: lc.LanguageClient,
    reqType: lc.RequestType<TParam, TRet, unknown>,
    param: TParam,
    token?: vscode.CancellationToken,
): Promise<TRet> {
    for (const delay of [2, 4, 6, 8, 10, null]) {
        try {
            return await (token
                ? client.sendRequest(reqType, param, token)
                : client.sendRequest(reqType, param)
            );
        } catch (error) {
            if (delay === null) {
                log.error("LSP request timed out", { method: reqType.method, param, error });
                throw error;
            }

            if (error.code === lc.ErrorCodes.RequestCancelled) {
                throw error;
            }

            if (error.code !== lc.ErrorCodes.ContentModified) {
                log.error("LSP request failed", { method: reqType.method, param, error });
                throw error;
            }

            await sleep(10 * (1 << delay));
        }
    }
    throw 'unreachable';
}

function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

export function notReentrant<TThis, TParams extends any[], TRet>(
    fn: (this: TThis, ...params: TParams) => Promise<TRet>
): typeof fn {
    let entered = false;
    return function(...params) {
        assert(!entered, `Reentrancy invariant for ${fn.name} is violated`);
        entered = true;
        return fn.apply(this, params).finally(() => entered = false);
    };
}

export type RustDocument = vscode.TextDocument & { languageId: "rust" };
export type RustEditor = vscode.TextEditor & { document: RustDocument; id: string };

export function isRustDocument(document: vscode.TextDocument): document is RustDocument {
    return document.languageId === 'rust'
        // SCM diff views have the same URI as the on-disk document but not the same content
        && document.uri.scheme !== 'git'
        && document.uri.scheme !== 'svn';
}

export function isRustEditor(editor: vscode.TextEditor): editor is RustEditor {
    return isRustDocument(editor.document);
}

/**
 * @param extensionId The canonical extension identifier in the form of: `publisher.name`
 */
export async function vscodeReinstallExtension(extensionId: string) {
    // Unfortunately there is no straightforward way as of now, these commands
    // were found in vscode source code.

    log.debug("Uninstalling extension", extensionId);
    await vscode.commands.executeCommand("workbench.extensions.uninstallExtension", extensionId);
    log.debug("Installing extension", extensionId);
    await vscode.commands.executeCommand("workbench.extensions.installExtension", extensionId);
}

export async function vscodeReloadWindow(): Promise<never> {
    await vscode.commands.executeCommand("workbench.action.reloadWindow");

    assert(false, "unreachable");
}

export async function vscodeInstallExtensionFromVsix(vsixPath: string) {
    await vscode.commands.executeCommand(
        "workbench.extensions.installExtension",
        vscode.Uri.file(vsixPath)
    );
}
