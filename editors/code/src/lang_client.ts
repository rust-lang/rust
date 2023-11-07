import * as lc from "vscode-languageclient/node";
import * as vscode from "vscode";

export class RaLanguageClient extends lc.LanguageClient {
    override error(message: string, data?: any, showNotification?: boolean | "force"): void {
        // ignore `Request TYPE failed.` errors
        const showError = vscode.workspace
            .getConfiguration("rust-analyzer")
            .get("showRequestFailedErrorNotification");
        if (!showError && message.startsWith("Request") && message.endsWith("failed.")) {
            return;
        }

        super.error(message, data, showNotification);
    }
}
