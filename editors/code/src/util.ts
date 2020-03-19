import * as lc from "vscode-languageclient";
import * as vscode from "vscode";
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
