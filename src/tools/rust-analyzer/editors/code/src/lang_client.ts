import * as lc from "vscode-languageclient/node";
import * as vscode from "vscode";

export class RaLanguageClient extends lc.LanguageClient {
    override handleFailedRequest<T>(
        type: lc.MessageSignature,
        token: vscode.CancellationToken | undefined,
        error: any,
        defaultValue: T,
        showNotification?: boolean | undefined,
    ): T {
        const showError = vscode.workspace
            .getConfiguration("rust-analyzer")
            .get("showRequestFailedErrorNotification");
        if (
            !showError &&
            error instanceof lc.ResponseError &&
            error.code === lc.ErrorCodes.InternalError
        ) {
            // Don't show notification for internal errors, these are emitted by r-a when a request fails.
            showNotification = false;
        }

        return super.handleFailedRequest(type, token, error, defaultValue, showNotification);
    }
}
