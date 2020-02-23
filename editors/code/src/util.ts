import * as lc from "vscode-languageclient";
import * as vscode from "vscode";

let enabled: boolean = false;

export const log = {
    debug(message?: any, ...optionalParams: any[]): void {
        if (!enabled) return;
        // eslint-disable-next-line no-console
        console.log(message, ...optionalParams);
    },
    error(message?: any, ...optionalParams: any[]): void {
        if (!enabled) return;
        debugger;
        // eslint-disable-next-line no-console
        console.error(message, ...optionalParams);
    },
    setEnabled(yes: boolean): void {
        enabled = yes;
    }
};

export async function sendRequestWithRetry<R>(
    client: lc.LanguageClient,
    method: string,
    param: unknown,
    token?: vscode.CancellationToken,
): Promise<R> {
    for (const delay of [2, 4, 6, 8, 10, null]) {
        try {
            return await (token
                ? client.sendRequest(method, param, token)
                : client.sendRequest(method, param)
            );
        } catch (error) {
            if (delay === null) {
                log.error("LSP request timed out", { method, param, error });
                throw error;
            }

            if (error.code === lc.ErrorCodes.RequestCancelled) {
                throw error;
            }

            if (error.code !== lc.ErrorCodes.ContentModified) {
                log.error("LSP request failed", { method, param, error });
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
