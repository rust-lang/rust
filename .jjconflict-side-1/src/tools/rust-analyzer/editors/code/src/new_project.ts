import * as vscode from "vscode";

import type { Ctx, Cmd } from "./ctx";
import * as ra from "./lsp_ext";
import { cargoPath } from "./toolchain";
import { log, spawnAsync } from "./util";

type NewProjectKind = "bin" | "lib";
type NewProjectOpenAction = "open" | "openNewWindow" | "addToWorkspace";

type NewProjectTemplate = {
    detail: string;
    id: NewProjectKind;
    label: string;
};

type NewProjectCargo = {
    cargo: string;
    cargoEnv: NodeJS.ProcessEnv;
};

const NEW_PROJECT_TEMPLATES: readonly NewProjectTemplate[] = [
    {
        id: "bin",
        label: "Binary Application",
        detail: "Create a Cargo binary package (`cargo new --bin`)",
    },
    {
        id: "lib",
        label: "Library",
        detail: "Create a Cargo library package (`cargo new --lib`)",
    },
] as const;

export function newProject(ctx: Ctx): Cmd {
    return async () => {
        const cargo = await resolveNewProjectCargo(ctx);

        const selectedKind = await promptForNewProjectTemplate();
        if (!selectedKind) {
            return;
        }

        const parentFolder = await promptForNewProjectParentFolder();
        if (!parentFolder) {
            return;
        }

        const projectName = await promptForNewProjectName(parentFolder);
        if (!projectName) {
            return;
        }

        if (!(await createNewProject(cargo, parentFolder, selectedKind, projectName))) {
            return;
        }

        const projectUri = vscode.Uri.joinPath(parentFolder, projectName);
        const defaultAction = determineNewProjectOpenAction(
            ctx.config.projectCreationOpenAfterCreate,
            Boolean(vscode.workspace.workspaceFolders?.length),
        );
        const action =
            defaultAction === "ask"
                ? await promptForNewProjectOpenAction(
                      projectName,
                      Boolean(vscode.workspace.workspaceFolders?.length),
                  )
                : defaultAction;

        if (action) {
            await executeNewProjectOpenAction(ctx, action, projectUri);
        }
    };
}

async function resolveNewProjectCargo(ctx: Ctx): Promise<NewProjectCargo> {
    // Use the same effective environment rust-analyzer uses elsewhere so project creation sees
    // toolchain wrappers, PATH overrides, and CARGO_HOME changes from configuration.
    const cargoEnv = { ...process.env, ...ctx.config.serverExtraEnv };
    return { cargo: await cargoPath(cargoEnv), cargoEnv };
}

async function promptForNewProjectTemplate(): Promise<NewProjectKind | undefined> {
    const selected = await vscode.window.showQuickPick(NEW_PROJECT_TEMPLATES, {
        placeHolder: "Select a Rust project kind",
    });
    return selected?.id;
}

async function promptForNewProjectParentFolder(): Promise<vscode.Uri | undefined> {
    const selectedFolder = await vscode.window.showOpenDialog({
        title: "Select the parent folder for the new Rust project",
        openLabel: "Select parent folder",
        canSelectFiles: false,
        canSelectFolders: true,
        canSelectMany: false,
    });
    if (!selectedFolder?.length) {
        return undefined;
    }
    return selectedFolder[0];
}

const CARGO_MANIFEST_NAME_PATTERN = /^[\p{Alphabetic}\p{Number}_-]+$/u;

// Keep local validation focused on stable checks that can be reported in the input box, then let
// `cargo new` remain the source of truth for package-name-specific identifier and keyword rules.
export function validateNewProjectName(
    value: string,
    existingNames: readonly string[],
): string | undefined {
    const trimmedValue = value.trim();
    if (trimmedValue.length === 0) {
        return "Project name cannot be empty.";
    }
    if (trimmedValue.includes("/") || trimmedValue.includes("\\")) {
        return "Project name cannot contain '/' or '\\' characters.";
    }
    if (trimmedValue === "." || trimmedValue === "..") {
        return "Project name cannot be '.' or '..'.";
    }
    if (!CARGO_MANIFEST_NAME_PATTERN.test(trimmedValue)) {
        return "Project name can contain only alphanumeric characters, '-' or '_'.";
    }
    if (existingNames.includes(trimmedValue)) {
        return "A file or folder with this name already exists.";
    }
    return undefined;
}

async function promptForNewProjectName(parentFolder: vscode.Uri): Promise<string | undefined> {
    let existingNames: string[] = [];
    try {
        const entries = await vscode.workspace.fs.readDirectory(parentFolder);
        existingNames = entries.map(([name]) => name);
    } catch (error) {
        log.error("Failed to read project parent folder", error);
        void vscode.window.showErrorMessage("Failed to read the selected parent folder.");
        return undefined;
    }

    const projectName = await vscode.window.showInputBox({
        prompt: `Enter the new project name to create inside ${parentFolder.fsPath}`,
        validateInput: async (value) => validateNewProjectName(value, existingNames),
    });
    return projectName?.trim();
}

export function cargoNewArgs(kind: NewProjectKind, name: string): string[] {
    return ["new", kind === "bin" ? "--bin" : "--lib", name];
}

async function createNewProject(
    cargo: NewProjectCargo,
    parentFolder: vscode.Uri,
    kind: NewProjectKind,
    projectName: string,
): Promise<boolean> {
    const args = cargoNewArgs(kind, projectName);
    const createResult = await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: `Creating Rust project ${projectName}`,
        },
        async () =>
            spawnAsync(cargo.cargo, args, {
                cwd: parentFolder.fsPath,
                env: cargo.cargoEnv,
            }),
    );

    if (createResult.error || createResult.status !== 0) {
        const details = formatProcessDetails(createResult);
        await showNewProjectError("Failed to create Rust project.", details || undefined, {
            cargo: cargo.cargo,
            args,
            cwd: parentFolder.fsPath,
            error: createResult.error?.message,
            status: createResult.status,
            stderr: createResult.stderr || undefined,
            stdout: createResult.stdout || undefined,
        });
        return false;
    }

    return true;
}

function formatProcessDetails(result: { error?: Error; stderr: string; stdout: string }): string {
    return [result.stderr, result.stdout, result.error?.message]
        .filter((value): value is string => Boolean(value && value.trim().length > 0))
        .join("\n")
        .trim();
}

async function showNewProjectError(
    message: string,
    details: string | undefined,
    logContext: {
        cargo: string;
        args: string[];
        cwd?: string;
        error?: string;
        status: number | null;
        stderr?: string;
        stdout?: string;
    },
): Promise<void> {
    // Keep command-failure logging focused on the invocation and process output. Environment
    // variables may contain secrets such as API keys, tokens, and credentials, so failure logs
    // must not dump the merged env here.
    const commandLine = [logContext.cargo, ...logContext.args].join(" ");
    log.error(message);
    log.error(`command: ${commandLine}`);
    if (logContext.cwd) {
        log.error(`cwd: ${logContext.cwd}`);
    }
    log.error(`exit status: ${String(logContext.status)}`);
    if (logContext.error) {
        log.error(`error: ${logContext.error}`);
    }
    if (logContext.stderr) {
        log.error(`stderr:\n${logContext.stderr}`);
    }
    if (logContext.stdout) {
        log.error(`stdout:\n${logContext.stdout}`);
    }
    const selection = await vscode.window.showErrorMessage(
        details ? `${message}\n${details}` : message,
        "Open Extension Logs",
    );
    if (selection === "Open Extension Logs") {
        log.show();
    }
}

export function determineNewProjectOpenAction(
    configuredAction: string | undefined,
    hasWorkspaceFolders: boolean,
): "ask" | NewProjectOpenAction {
    switch (configuredAction) {
        case "open":
        case "openNewWindow":
            return configuredAction;
        case "addToWorkspace":
            // Adding to a workspace only makes sense when one is already open. Falling back to
            // "open" keeps the setting usable in empty windows without adding another prompt path.
            return hasWorkspaceFolders ? configuredAction : "open";
        default:
            return "ask";
    }
}

async function promptForNewProjectOpenAction(
    projectName: string,
    hasWorkspaceFolders: boolean,
): Promise<NewProjectOpenAction | undefined> {
    let message = `Would you like to open ${projectName}?`;
    const open = "Open";
    const openNewWindow = "Open in New Window";
    const choices = [open, openNewWindow];

    const addToWorkspace = "Add to VS Code Workspace";
    if (hasWorkspaceFolders) {
        message = `Would you like to open ${projectName}, or add it to the current VS Code workspace?`;
        choices.push(addToWorkspace);
    }

    const result = await vscode.window.showInformationMessage(
        message,
        { modal: true, detail: "The default action can be configured in settings." },
        ...choices,
    );

    const actionMap: Record<string, NewProjectOpenAction> = {
        [open]: "open",
        [openNewWindow]: "openNewWindow",
        [addToWorkspace]: "addToWorkspace",
    };
    return result ? actionMap[result] : undefined;
}

async function executeNewProjectOpenAction(
    ctx: Ctx,
    action: NewProjectOpenAction,
    projectUri: vscode.Uri,
): Promise<void> {
    if (action === "open") {
        await vscode.commands.executeCommand("vscode.openFolder", projectUri, {
            forceReuseWindow: true,
        });
        return;
    }

    if (action === "openNewWindow") {
        await vscode.commands.executeCommand("vscode.openFolder", projectUri, {
            forceNewWindow: true,
        });
        return;
    }

    const index = vscode.workspace.workspaceFolders?.length ?? 0;
    vscode.workspace.updateWorkspaceFolders(index, 0, { uri: projectUri });
    // Reuse the existing workspace window when requested, but nudge rust-analyzer afterwards so
    // the newly added Cargo project is discovered immediately instead of waiting for a later
    // background refresh.
    if (ctx.client?.isRunning()) {
        await ctx.client.sendRequest(ra.reloadWorkspace);
    }
}
