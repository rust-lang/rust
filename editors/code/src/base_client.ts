import * as lc from "vscode-languageclient/node";

export class RaLanguageClient extends lc.LanguageClient {
    override error(message: string, data?: any, showNotification?: boolean | "force"): void {
        // ignore `Request TYPE failed.` errors
        if (message.startsWith("Request") && message.endsWith("failed.")) {
            return;
        }

        super.error(message, data, showNotification);
    }
}
