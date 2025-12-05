import * as vscode from "vscode";
import { strict as nativeAssert } from "assert";
import { exec, spawn, type SpawnOptionsWithoutStdio, type ExecOptions } from "child_process";
import { inspect } from "util";
import type { CargoRunnableArgs, ShellRunnableArgs } from "./lsp_ext";

export function assert(condition: boolean, explanation: string): asserts condition {
    try {
        nativeAssert(condition, explanation);
    } catch (err) {
        log.error(`Assertion failed:`, explanation);
        throw err;
    }
}

export type Env = {
    [name: string]: string | undefined;
};

class Log {
    private readonly output = vscode.window.createOutputChannel("rust-analyzer Extension", {
        log: true,
    });

    trace(...messages: [unknown, ...unknown[]]): void {
        this.output.trace(this.stringify(messages));
    }

    debug(...messages: [unknown, ...unknown[]]): void {
        this.output.debug(this.stringify(messages));
    }

    info(...messages: [unknown, ...unknown[]]): void {
        this.output.info(this.stringify(messages));
    }

    warn(...messages: [unknown, ...unknown[]]): void {
        this.output.warn(this.stringify(messages));
    }

    error(...messages: [unknown, ...unknown[]]): void {
        this.output.error(this.stringify(messages));
        this.output.show(true);
    }

    private stringify(messages: unknown[]): string {
        return messages
            .map((message) => {
                if (typeof message === "string") {
                    return message;
                }
                if (message instanceof Error) {
                    return message.stack || message.message;
                }
                return inspect(message, { depth: 6, colors: false });
            })
            .join(" ");
    }
}

export const log = new Log();

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

export function isCargoRunnableArgs(
    args: CargoRunnableArgs | ShellRunnableArgs,
): args is CargoRunnableArgs {
    return (args as CargoRunnableArgs).executableArgs !== undefined;
}

export function isRustEditor(editor: vscode.TextEditor): editor is RustEditor {
    return isRustDocument(editor.document);
}

export function isCargoTomlEditor(editor: vscode.TextEditor): editor is RustEditor {
    return isCargoTomlDocument(editor.document);
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

/** Sets ['when'](https://code.visualstudio.com/docs/getstarted/keybindings#_when-clause-contexts) clause contexts */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function setContextValue(key: string, value: any): Thenable<void> {
    return vscode.commands.executeCommand("setContext", key, value);
}

/**
 * Returns a higher-order function that caches the results of invoking the
 * underlying async function.
 */
export function memoizeAsync<Ret, TThis, Param extends string>(
    func: (this: TThis, arg: Param) => Promise<Ret>,
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
                log.error("error:", err);
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

    show(columnOrPreserveFocus?: vscode.ViewColumn | boolean, preserveFocus?: boolean): void {
        if (typeof columnOrPreserveFocus === "boolean") {
            this.channel.show(columnOrPreserveFocus);
        } else {
            this.channel.show(columnOrPreserveFocus, preserveFocus);
        }
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

export type NotNull<T> = T extends null ? never : T;

export type Nullable<T> = T | null;

function isNotNull<T>(input: Nullable<T>): input is NotNull<T> {
    return input !== null;
}

function expectNotNull<T>(input: Nullable<T>, msg: string): NotNull<T> {
    if (isNotNull(input)) {
        return input;
    }

    throw new TypeError(msg);
}

export function unwrapNullable<T>(input: Nullable<T>): NotNull<T> {
    return expectNotNull(input, `unwrapping \`null\``);
}
export type NotUndefined<T> = T extends undefined ? never : T;

export type Undefinable<T> = T | undefined;

function isNotUndefined<T>(input: Undefinable<T>): input is NotUndefined<T> {
    return input !== undefined;
}

export function expectNotUndefined<T>(input: Undefinable<T>, msg: string): NotUndefined<T> {
    if (isNotUndefined(input)) {
        return input;
    }

    throw new TypeError(msg);
}

export function unwrapUndefinable<T>(input: Undefinable<T>): NotUndefined<T> {
    return expectNotUndefined(input, `unwrapping \`undefined\``);
}

interface SpawnAsyncReturns {
    stdout: string;
    stderr: string;
    status: number | null;
    error?: Error | undefined;
}

export async function spawnAsync(
    path: string,
    args?: ReadonlyArray<string>,
    options?: SpawnOptionsWithoutStdio,
): Promise<SpawnAsyncReturns> {
    const child = spawn(path, args, options);
    const stdout: Array<Buffer> = [];
    const stderr: Array<Buffer> = [];
    try {
        const res = await new Promise<{ stdout: string; stderr: string; status: number | null }>(
            (resolve, reject) => {
                child.stdout.on("data", (chunk) => stdout.push(Buffer.from(chunk)));
                child.stderr.on("data", (chunk) => stderr.push(Buffer.from(chunk)));
                child.on("error", (error) =>
                    reject({
                        stdout: Buffer.concat(stdout).toString("utf8"),
                        stderr: Buffer.concat(stderr).toString("utf8"),
                        error,
                    }),
                );
                child.on("close", (status) =>
                    resolve({
                        stdout: Buffer.concat(stdout).toString("utf8"),
                        stderr: Buffer.concat(stderr).toString("utf8"),
                        status,
                    }),
                );
            },
        );

        return {
            stdout: res.stdout,
            stderr: res.stderr,
            status: res.status,
        };
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (e: any) {
        return {
            stdout: e.stdout,
            stderr: e.stderr,
            status: e.status,
            error: e.error,
        };
    }
}

export const isWindows = process.platform === "win32";

export function isWindowsDriveLetter(code: number): boolean {
    // Copied from https://github.com/microsoft/vscode/blob/02c2dba5f2669b924fd290dff7d2ff3460791996/src/vs/base/common/extpath.ts#L265-L267
    return (
        (code >= /* CharCode.A */ 65 && code <= /* CharCode.Z */ 90) ||
        (code >= /* CharCode.a */ 97 && code <= /* CharCode.z */ 122)
    );
}
export function hasDriveLetter(path: string, isWindowsOS: boolean = isWindows): boolean {
    // Copied from https://github.com/microsoft/vscode/blob/02c2dba5f2669b924fd290dff7d2ff3460791996/src/vs/base/common/extpath.ts#L324-L330
    if (isWindowsOS) {
        return (
            isWindowsDriveLetter(path.charCodeAt(0)) &&
            path.charCodeAt(1) === /* CharCode.Colon */ 58
        );
    }

    return false;
}
export function normalizeDriveLetter(path: string, isWindowsOS: boolean = isWindows): string {
    // Copied from https://github.com/microsoft/vscode/blob/02c2dba5f2669b924fd290dff7d2ff3460791996/src/vs/base/common/labels.ts#L140-L146
    if (hasDriveLetter(path, isWindowsOS)) {
        return path.charAt(0).toUpperCase() + path.slice(1);
    }

    return path;
}
