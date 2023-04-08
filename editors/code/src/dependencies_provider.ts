import * as vscode from "vscode";
import * as fspath from "path";
import * as fs from "fs";
import { CtxInit } from "./ctx";
import * as ra from "./lsp_ext";
import { FetchDependencyListResult } from "./lsp_ext";
import { Ctx } from "./ctx";
import { setFlagsFromString } from "v8";
import * as ra from "./lsp_ext";


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

    refresh(): void {
        this.dependenciesMap = {};
        this._onDidChangeTreeData.fire();
    }

    getParent?(
        element: Dependency | DependencyFile
    ): vscode.ProviderResult<Dependency | DependencyFile> {
        if (element instanceof Dependency) return undefined;
        return element.parent;
    }

    getTreeItem(element: Dependency | DependencyFile): vscode.TreeItem | Thenable<vscode.TreeItem> {
        if (element.id! in this.dependenciesMap) return this.dependenciesMap[element.id!];
        return element;
    }

    getChildren(
        element?: Dependency | DependencyFile
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
        const crates = await this.ctx.client.sendRequest(ra.fetchDependencyGraph, {});

        const dependenciesResult: FetchDependencyListResult = await this.ctx.client.sendRequest(
            ra.fetchDependencyList,
            {}
        );
        const crates = dependenciesResult.crates;
        const deps = crates.map((crate) => {
        const dep = this.toDep(crate.name, crate.version, crate.path);
            this.dependenciesMap[dep.dependencyPath.toLowerCase()] = dep;
        this.dependenciesMap[stdlib.dependencyPath.toLowerCase()] = stdlib;
        return dep;
        });
        return deps;
    }

    private toDep(moduleName: string, version: string, path: string): Dependency {
        return new Dependency(moduleName, version, path, vscode.TreeItemCollapsibleState.Collapsed);
    }
}

export class Dependency extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        private version: string,
        readonly dependencyPath: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(label, collapsibleState);
        this.id = this.dependencyPath.toLowerCase();
        this.description = this.version;
        this.resourceUri = vscode.Uri.file(dependencyPath);
        if (this.version) {
            this.tooltip = `${this.label}-${this.version}`;
        } else {
            this.tooltip = this.label;
        }
    }
}

export class DependencyFile extends vscode.TreeItem {
    constructor(
        readonly label: string,
        readonly dependencyPath: string,
        readonly parent: Dependency | DependencyFile,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(vscode.Uri.file(dependencyPath), collapsibleState);
        this.id = this.dependencyPath.toLowerCase();
        const isDir = fs.lstatSync(this.dependencyPath).isDirectory();
        if (!isDir) {
            this.command = { command: "vscode.open",
                title: "Open File",
                arguments: [vscode.Uri.file(this.dependencyPath)],
        };
    }}
}

export type DependencyId = { id: string };
