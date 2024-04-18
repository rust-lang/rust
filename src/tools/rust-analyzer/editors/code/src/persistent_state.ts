import type * as vscode from "vscode";
import { log } from "./util";

export class PersistentState {
    constructor(private readonly globalState: vscode.Memento) {
        const { serverVersion } = this;
        log.info("PersistentState:", { serverVersion });
    }

    /**
     * Version of the extension that installed the server.
     * Used to check if we need to run patchelf again on NixOS.
     */
    get serverVersion(): string | undefined {
        return this.globalState.get("serverVersion");
    }
    async updateServerVersion(value: string | undefined) {
        await this.globalState.update("serverVersion", value);
    }
}
