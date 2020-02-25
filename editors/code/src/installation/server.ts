import * as vscode from "vscode";
import * as path from "path";
import { strict as assert } from "assert";
import { promises as dns } from "dns";
import { spawnSync } from "child_process";

import { BinarySource } from "./interfaces";
import { fetchArtifactReleaseInfo } from "./fetch_artifact_release_info";
import { downloadArtifact } from "./download_artifact";
import { log } from "../util";

export async function ensureServerBinary(source: null | BinarySource): Promise<null | string> {
    if (!source) {
        vscode.window.showErrorMessage(
            "Unfortunately we don't ship binaries for your platform yet. " +
            "You need to manually clone rust-analyzer repository and " +
            "run `cargo xtask install --server` to build the language server from sources. " +
            "If you feel that your platform should be supported, please create an issue " +
            "about that [here](https://github.com/rust-analyzer/rust-analyzer/issues) and we " +
            "will consider it."
        );
        return null;
    }

    switch (source.type) {
        case BinarySource.Type.ExplicitPath: {
            if (isBinaryAvailable(source.path)) {
                return source.path;
            }

            vscode.window.showErrorMessage(
                `Unable to run ${source.path} binary. ` +
                `To use the pre-built language server, set "rust-analyzer.serverPath" ` +
                "value to `null` or remove it from the settings to use it by default."
            );
            return null;
        }
        case BinarySource.Type.GithubRelease: {
            const prebuiltBinaryPath = path.join(source.dir, source.file);

            const installedVersion: null | string = getServerVersion(source.storage);
            const requiredVersion: string = source.version;

            log.debug("Installed version:", installedVersion, "required:", requiredVersion);

            if (isBinaryAvailable(prebuiltBinaryPath) && installedVersion === requiredVersion) {
                return prebuiltBinaryPath;
            }

            const userResponse = await vscode.window.showInformationMessage(
                `Language server version ${source.version} for rust-analyzer is not installed. ` +
                "Do you want to download it now?",
                "Download now", "Cancel"
            );
            if (userResponse !== "Download now") return null;

            if (!await downloadServer(source)) return null;

            return prebuiltBinaryPath;
        }
    }
}

async function downloadServer(source: BinarySource.GithubRelease): Promise<boolean> {
    try {
        const releaseInfo = await fetchArtifactReleaseInfo(source.repo, source.file, source.version);

        await downloadArtifact(releaseInfo, source.file, source.dir, "language server");
        await setServerVersion(source.storage, releaseInfo.releaseName);
    } catch (err) {
        vscode.window.showErrorMessage(
            `Failed to download language server from ${source.repo.name} ` +
            `GitHub repository: ${err.message}`
        );

        log.error(err);

        dns.resolve('example.com').then(
            addrs => log.debug("DNS resolution for example.com was successful", addrs),
            err => {
                log.error(
                    "DNS resolution for example.com failed, " +
                    "there might be an issue with Internet availability"
                );
                log.error(err);
            }
        );
        return false;
    }

    const binaryPath = path.join(source.dir, source.file);

    assert(isBinaryAvailable(binaryPath),
        `Downloaded language server binary is not functional.` +
        `Downloaded from GitHub repo ${source.repo.owner}/${source.repo.name} ` +
        `to ${binaryPath}`
    );

    vscode.window.showInformationMessage(
        "Rust analyzer language server was successfully installed 🦀"
    );

    return true;
}

function isBinaryAvailable(binaryPath: string): boolean {
    const res = spawnSync(binaryPath, ["--version"]);

    // ACHTUNG! `res` type declaration is inherently wrong, see
    // https://github.com/DefinitelyTyped/DefinitelyTyped/issues/42221

    log.debug("Checked binary availablity via --version", res);
    log.debug(binaryPath, "--version output:", res.output?.map(String));

    return res.status === 0;
}

function getServerVersion(storage: vscode.Memento): null | string {
    const version = storage.get<null | string>("server-version", null);
    log.debug("Get server-version:", version);
    return version;
}

async function setServerVersion(storage: vscode.Memento, version: string): Promise<void> {
    log.debug("Set server-version:", version);
    await storage.update("server-version", version.toString());
}
