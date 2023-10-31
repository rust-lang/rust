import * as vscode from "vscode";
import * as fspath from "path";
import * as fs from "fs";
import type { CtxInit } from "./ctx";
import * as ra from "./lsp_ext";
import type { FetchDependencyListResult } from "./lsp_ext";
import { unwrapUndefinable } from "./undefinable";

export class RustDependenciesProvider
    implements vscode.TreeDataProvider<Dependency | DependencyFile>
{
    dependenciesMap: { [id: string]: Dependency | DependencyFile };
    ctx: CtxInit;

    constructor(ctx: CtxInit) {
        this.dependenciesMap = {};
        this.ctx = ctx;
    }

    private _onDidChangeTreeData: vscode.EventEmitter<
        Dependency | DependencyFile | undefined | null | void
    > = new vscode.EventEmitter<Dependency | undefined | null | void>();

    readonly onDidChangeTreeData: vscode.Event<
        Dependency | DependencyFile | undefined | null | void
    > = this._onDidChangeTreeData.event;

    getDependency(filePath: string): Dependency | DependencyFile | undefined {
        return this.dependenciesMap[filePath.toLowerCase()];
    }

    contains(filePath: string): boolean {
        return filePath.toLowerCase() in this.dependenciesMap;
    }

    isInitialized(): boolean {
        return Object.keys(this.dependenciesMap).length !== 0;
    }

    refresh(): void {
        this.dependenciesMap = {};
        this._onDidChangeTreeData.fire();
    }

    getParent?(
        element: Dependency | DependencyFile,
    ): vscode.ProviderResult<Dependency | DependencyFile> {
        if (element instanceof Dependency) return undefined;
        return element.parent;
    }

    getTreeItem(element: Dependency | DependencyFile): vscode.TreeItem | Thenable<vscode.TreeItem> {
        const dependenciesMap = this.dependenciesMap;
        const elementId = element.id!;
        if (elementId in dependenciesMap) {
            const dependency = unwrapUndefinable(dependenciesMap[elementId]);
            return dependency;
        }
        return element;
    }

    getChildren(
        element?: Dependency | DependencyFile,
    ): vscode.ProviderResult<Dependency[] | DependencyFile[]> {
        return new Promise((resolve, _reject) => {
            if (!vscode.workspace.workspaceFolders) {
                void vscode.window.showInformationMessage("No dependency in empty workspace");
                return Promise.resolve([]);
            }
            if (element) {
                const files = fs.readdirSync(element.dependencyPath).map((fileName) => {
                    const filePath = fspath.join(element.dependencyPath, fileName);
                    const collapsibleState = fs.lstatSync(filePath).isDirectory()
                        ? vscode.TreeItemCollapsibleState.Collapsed
                        : vscode.TreeItemCollapsibleState.None;
                    const dep = new DependencyFile(fileName, filePath, element, collapsibleState);
                    this.dependenciesMap[dep.dependencyPath.toLowerCase()] = dep;
                    return dep;
                });
                return resolve(files);
            } else {
                return resolve(this.getRootDependencies());
            }
        });
    }

    private async getRootDependencies(): Promise<Dependency[]> {
        const dependenciesResult: FetchDependencyListResult = await this.ctx.client.sendRequest(
            ra.fetchDependencyList,
            {},
        );
        const crates = dependenciesResult.crates;

        return crates
            .map((crate) => {
                const dep = this.toDep(crate.name || "unknown", crate.version || "", crate.path);
                this.dependenciesMap[dep.dependencyPath.toLowerCase()] = dep;
                return dep;
            })
            .sort((a, b) => {
                return a.label.localeCompare(b.label);
            });
    }

    private toDep(moduleName: string, version: string, path: string): Dependency {
        return new Dependency(
            moduleName,
            version,
            vscode.Uri.parse(path).fsPath,
            vscode.TreeItemCollapsibleState.Collapsed,
        );
    }
}

export class Dependency extends vscode.TreeItem {
    constructor(
        public override readonly label: string,
        private version: string,
        readonly dependencyPath: string,
        public override readonly collapsibleState: vscode.TreeItemCollapsibleState,
    ) {
        super(label, collapsibleState);
        this.resourceUri = vscode.Uri.file(dependencyPath);
        this.id = this.resourceUri.fsPath.toLowerCase();
        this.description = this.version;
        if (this.version) {
            this.tooltip = `${this.label}-${this.version}`;
        } else {
            this.tooltip = this.label;
        }
    }
}

export class DependencyFile extends vscode.TreeItem {
    constructor(
        override readonly label: string,
        readonly dependencyPath: string,
        readonly parent: Dependency | DependencyFile,
        public override readonly collapsibleState: vscode.TreeItemCollapsibleState,
    ) {
        super(vscode.Uri.file(dependencyPath), collapsibleState);
        this.id = this.resourceUri!.fsPath.toLowerCase();
        const isDir = fs.lstatSync(this.resourceUri!.fsPath).isDirectory();
        if (!isDir) {
            this.command = {
                command: "vscode.open",
                title: "Open File",
                arguments: [this.resourceUri],
            };
        }
    }
}

export type DependencyId = { id: string };
