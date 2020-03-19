import * as vscode from 'vscode';
import { log } from './util';

export class PersistentState {
    constructor(private readonly globalState: vscode.Memento) {
        const { lastCheck, releaseId, serverVersion } = this;
        log.debug("PersistentState: ", { lastCheck, releaseId, serverVersion });
    }

    /**
     * Used to check for *nightly* updates once an hour.
     */
    get lastCheck(): number | undefined {
        return this.globalState.get("lastCheck");
    }
    async updateLastCheck(value: number) {
        await this.globalState.update("lastCheck", value);
    }

    /**
     * Release id of the *nightly* extension.
     * Used to check if we should update.
     */
    get releaseId(): number | undefined {
        return this.globalState.get("releaseId");
    }
    async updateReleaseId(value: number) {
        await this.globalState.update("releaseId", value);
    }

    /**
     * Version of the extension that installed the server.
     * Used to check if we need to update the server.
     */
    get serverVersion(): string | undefined {
        return this.globalState.get("serverVersion");
    }
    async updateServerVersion(value: string | undefined) {
        await this.globalState.update("serverVersion", value);
    }
}
