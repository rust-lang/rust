import * as vscode from 'vscode';
import * as fspath from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { activeToolchain, Cargo, Crate, getRustcVersion } from './toolchain';

const debugOutput = vscode.window.createOutputChannel("Debug");

export class RustDependenciesProvider implements vscode.TreeDataProvider<Dependency | DependencyFile>{
  cargo: Cargo;
  dependenciesMap: { [id: string]: Dependency | DependencyFile };

  constructor(
    private readonly workspaceRoot: string,
  ) {
    this.cargo = new Cargo(this.workspaceRoot || '.', debugOutput);
    this.dependenciesMap = {};
  }

  private _onDidChangeTreeData: vscode.EventEmitter<Dependency | DependencyFile | undefined | null | void> = new vscode.EventEmitter<Dependency | undefined | null | void>();

  readonly onDidChangeTreeData: vscode.Event<Dependency | DependencyFile | undefined | null | void> = this._onDidChangeTreeData.event;


  getDependency(filePath: string): Dependency | DependencyFile | undefined {
    return this.dependenciesMap[filePath.toLowerCase()];
  }

  contains(filePath: string): boolean {
    return filePath.toLowerCase() in this.dependenciesMap;
  }

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getParent?(element: Dependency | DependencyFile): vscode.ProviderResult<Dependency | DependencyFile> {
    if (element instanceof Dependency) return undefined;
    return element.parent;
  }

  getTreeItem(element: Dependency | DependencyFile): vscode.TreeItem | Thenable<vscode.TreeItem> {
    if (element.id! in this.dependenciesMap) return this.dependenciesMap[element.id!];
    return element;
  }

  getChildren(element?: Dependency | DependencyFile): vscode.ProviderResult<Dependency[] | DependencyFile[]> {
    return new Promise((resolve, _reject) => {
      if (!this.workspaceRoot) {
        void vscode.window.showInformationMessage('No dependency in empty workspace');
        return Promise.resolve([]);
      }

      if (element) {
        const files = fs.readdirSync(element.dependencyPath).map(fileName => {
          const filePath = fspath.join(element.dependencyPath, fileName);
          const collapsibleState = fs.lstatSync(filePath).isDirectory() ?
            vscode.TreeItemCollapsibleState.Collapsed :
            vscode.TreeItemCollapsibleState.None;
          const dep = new DependencyFile(
            fileName,
            filePath,
            element,
            collapsibleState
          );
          this.dependenciesMap[dep.dependencyPath.toLowerCase()] = dep;
          return dep;
        });
        return resolve(
          files
        );
      } else {
        return resolve(this.getRootDependencies());
      }
    });
  }

  private async getRootDependencies(): Promise<Dependency[]> {
    const registryDir = fspath.join(os.homedir(), '.cargo', 'registry', 'src');
    const basePath = fspath.join(registryDir, fs.readdirSync(registryDir)[0]);
    const deps = await this.getDepsInCartoTree(basePath);
    const stdlib = await this.getStdLib();
    return [stdlib].concat(deps);
  }

  private async getStdLib(): Promise<Dependency> {
    const toolchain = await activeToolchain();
    const rustVersion = await getRustcVersion(os.homedir());
    const stdlibPath = fspath.join(os.homedir(), '.rustup', 'toolchains', toolchain, 'lib', 'rustlib', 'src', 'rust', 'library');
    return new Dependency(
      "stdlib",
      rustVersion,
      stdlibPath,
      vscode.TreeItemCollapsibleState.Collapsed
    );
  }

  private async getDepsInCartoTree(basePath: string): Promise<Dependency[]> {
    const crates: Crate[] = await this.cargo.crates();
    const toDep = (moduleName: string, version: string): Dependency => {
      const cratePath = fspath.join(basePath, `${moduleName}-${version}`);
      return new Dependency(
        moduleName,
        version,
        cratePath,
        vscode.TreeItemCollapsibleState.Collapsed
      );
    };

    const deps = crates.map(crate => {
      const dep = toDep(crate.name, crate.version);
      this.dependenciesMap[dep.dependencyPath.toLowerCase()] = dep;
      return dep;
    });
    return deps;
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
    this.tooltip = `${this.label}-${this.version}`;
    this.description = this.version;
    this.resourceUri = vscode.Uri.file(dependencyPath);
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
    const isDir = fs.lstatSync(this.dependencyPath).isDirectory();
    this.id = this.dependencyPath.toLowerCase();
    if (!isDir) {
      this.command = { command: 'rust-analyzer.openFile', title: "Open File", arguments: [vscode.Uri.file(this.dependencyPath)], };
    }
  }
}

export type DependencyId = { id: string };