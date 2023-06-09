import * as lc from "vscode-languageclient/node";
import * as vscode from "vscode";
import { strict as nativeAssert } from "assert";
import { exec, ExecOptions, spawnSync } from "child_process";
import { inspect } from "util";

export function assert(condition: boolean, explanation: string): asserts condition {
    try {
        nativeAssert(condition, explanation);
    } catch (err) {
        log.error(`Assertion failed:`, explanation);
        throw err;
    }
}

export const log = new (class {
    private enabled = true;
    private readonly output = vscode.window.createOutputChannel("Rust Analyzer Client");

    setEnabled(yes: boolean): void {
        log.enabled = yes;
    }

    // Hint: the type [T, ...T[]] means a non-empty array
    debug(...msg: [unknown, ...unknown[]]): void {
        if (!log.enabled) return;
        log.write("DEBUG", ...msg);
    }

    info(...msg: [unknown, ...unknown[]]): void {
        log.write("INFO", ...msg);
    }

    warn(...msg: [unknown, ...unknown[]]): void {
        debugger;
        log.write("WARN", ...msg);
    }

    error(...msg: [unknown, ...unknown[]]): void {
        debugger;
        log.write("ERROR", ...msg);
        log.output.show(true);
    }

    private write(label: string, ...messageParts: unknown[]): void {
        const message = messageParts.map(log.stringify).join(" ");
        const dateTime = new Date().toLocaleString();
        log.output.appendLine(`${label} [${dateTime}]: ${message}`);
    }

    private stringify(val: unknown): string {
        if (typeof val === "string") return val;
        return inspect(val, {
            colors: false,
            depth: 6, // heuristic
        });
    }
})();

export async function sendRequestWithRetry<TParam, TRet>(
    client: lc.LanguageClient,
    reqType: lc.RequestType<TParam, TRet, unknown>,
    param: TParam,
    token?: vscode.CancellationToken
): Promise<TRet> {
    // The sequence is `10 * (2 ** (2 * n))` where n is 1, 2, 3...
    for (const delay of [40, 160, 640, 2560, 10240, null]) {
        try {
            return await (token
                ? client.sendRequest(reqType, param, token)
                : client.sendRequest(reqType, param));
        } catch (error) {
            if (delay === null) {
                log.warn("LSP request timed out", { method: reqType.method, param, error });
                throw error;
            }
            if (error.code === lc.LSPErrorCodes.RequestCancelled) {
                throw error;
            }

            if (error.code !== lc.LSPErrorCodes.ContentModified) {
                log.warn("LSP request failed", { method: reqType.method, param, error });
                throw error;
            }
            await sleep(delay);
        }
    }
    throw "unreachable";
}

export function sleep(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

export type RustDocument = vscode.TextDocument & { languageId: "rust" };
export type RustEditor = vscode.TextEditor & { document: RustDocument };

export function isRustDocument(document: vscode.TextDocument): document is RustDocument {
    // Prevent corrupted text (particularly via inlay hints) in diff views
    // by allowing only `file` schemes
    // unfortunately extensions that use diff views not always set this
    // to something different than 'file' (see ongoing bug: #4608)
    return document.languageId === "rust" && document.uri.scheme === "file";
}

export function isCargoTomlDocument(document: vscode.TextDocument): document is RustDocument {
    // ideally `document.languageId` should be 'toml' but user maybe not have toml extension installed
    return document.uri.scheme === "file" && document.fileName.endsWith("Cargo.toml");
}

export function isRustEditor(editor: vscode.TextEditor): editor is RustEditor {
    return isRustDocument(editor.document);
}

export function isDocumentInWorkspace(document: RustDocument): boolean {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
        return false;
    }
    for (const folder of workspaceFolders) {
        if (document.uri.fsPath.startsWith(folder.uri.fsPath)) {
            return true;
        }
    }
    return false;
}

export function isValidExecutable(path: string): boolean {
    log.debug("Checking availability of a binary at", path);

    const res = spawnSync(path, ["--version"], { encoding: "utf8" });

    const printOutput = res.error ? log.warn : log.info;
    printOutput(path, "--version:", res);

    return res.status === 0;
}

/** Sets ['when'](https://code.visualstudio.com/docs/getstarted/keybindings#_when-clause-contexts) clause contexts */
export function setContextValue(key: string, value: any): Thenable<void> {
    return vscode.commands.executeCommand("setContext", key, value);
}

/**
 * Returns a higher-order function that caches the results of invoking the
 * underlying async function.
 */
export function memoizeAsync<Ret, TThis, Param extends string>(
    func: (this: TThis, arg: Param) => Promise<Ret>
) {
    const cache = new Map<string, Ret>();

    return async function (this: TThis, arg: Param) {
        const cached = cache.get(arg);
        if (cached) return cached;

        const result = await func.call(this, arg);
        cache.set(arg, result);

        return result;
    };
}

/** Awaitable wrapper around `child_process.exec` */
export function execute(command: string, options: ExecOptions): Promise<string> {
    log.info(`running command: ${command}`);
    return new Promise((resolve, reject) => {
        exec(command, options, (err, stdout, stderr) => {
            if (err) {
                log.error(err);
                reject(err);
                return;
            }

            if (stderr) {
                reject(new Error(stderr));
                return;
            }

            resolve(stdout.trimEnd());
        });
    });
}

export function executeDiscoverProject(command: string, options: ExecOptions): Promise<string> {
    log.info(`running command: ${command}`);
    return new Promise((resolve, reject) => {
        exec(command, options, (err, stdout, _) => {
            if (err) {
                log.error(err);
                reject(err);
                return;
            }

            resolve(stdout.trimEnd());
        });
    });
}

export class LazyOutputChannel implements vscode.OutputChannel {
    constructor(name: string) {
        this.name = name;
    }

    name: string;
    _channel: vscode.OutputChannel | undefined;

    get channel(): vscode.OutputChannel {
        if (!this._channel) {
            this._channel = vscode.window.createOutputChannel(this.name);
        }
        return this._channel;
    }

    append(value: string): void {
        this.channel.append(value);
    }
    appendLine(value: string): void {
        this.channel.appendLine(value);
    }
    replace(value: string): void {
        this.channel.replace(value);
    }
    clear(): void {
        if (this._channel) {
            this._channel.clear();
        }
    }
    show(preserveFocus?: boolean): void;
    show(column?: vscode.ViewColumn, preserveFocus?: boolean): void;
    show(column?: any, preserveFocus?: any): void {
        this.channel.show(column, preserveFocus);
    }
    hide(): void {
        if (this._channel) {
            this._channel.hide();
        }
    }
    dispose(): void {
        if (this._channel) {
            this._channel.dispose();
        }
    }
}
