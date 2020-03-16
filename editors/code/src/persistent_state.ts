import * as vscode from 'vscode';
import { log } from "./util";

export class PersistentState {
    constructor(private readonly ctx: vscode.ExtensionContext) {
    }

    readonly installedNightlyExtensionReleaseDate = new DateStorage(
        "installed-nightly-extension-release-date",
        this.ctx.globalState
    );
    readonly serverReleaseDate = new DateStorage("server-release-date", this.ctx.globalState);
    readonly serverReleaseTag = new Storage<null | string>("server-release-tag", this.ctx.globalState, null);
}


export class Storage<T> {
    constructor(
        private readonly key: string,
        private readonly storage: vscode.Memento,
        private readonly defaultVal: T
    ) { }

    get(): T {
        const val = this.storage.get(this.key, this.defaultVal);
        log.debug(this.key, "==", val);
        return val;
    }
    async set(val: T) {
        log.debug(this.key, "=", val);
        await this.storage.update(this.key, val);
    }
}
export class DateStorage {
    inner: Storage<null | string>;

    constructor(key: string, storage: vscode.Memento) {
        this.inner = new Storage(key, storage, null);
    }

    get(): null | Date {
        const dateStr = this.inner.get();
        return dateStr ? new Date(dateStr) : null;
    }

    async set(date: null | Date) {
        await this.inner.set(date ? date.toString() : null);
    }
}
