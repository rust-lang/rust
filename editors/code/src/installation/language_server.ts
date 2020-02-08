import { unwrapNotNil } from "ts-not-nil";
import { spawnSync } from "child_process";
import * as vscode from "vscode";
import * as path from "path";
import { strict as assert } from "assert";
import { promises as fs } from "fs";

import { BinarySource, BinarySourceType, GithubBinarySource } from "./interfaces";
import { fetchLatestArtifactMetadata } from "./fetch_latest_artifact_metadata";
import { downloadFile } from "./download_file";

export async function downloadLatestLanguageServer(
    {file: artifactFileName, dir: installationDir, repo}: GithubBinarySource
) {
    const binaryMetadata = await fetchLatestArtifactMetadata({ artifactFileName, repo });

    const {
        releaseName,
        downloadUrl
    } = unwrapNotNil(binaryMetadata, `Latest GitHub release lacks "${artifactFileName}" file`);

    await fs.mkdir(installationDir).catch(err => assert.strictEqual(
        err && err.code,
        "EEXIST",
        `Couldn't create directory "${installationDir}" to download `+
        `language server binary: ${err.message}`
    ));

    const installationPath = path.join(installationDir, artifactFileName);

    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            cancellable: false, // FIXME: add support for canceling download?
            title: `Downloading language server ${releaseName}`
        },
        async (progress, _) => {
            let lastPrecentage = 0;
            await downloadFile(downloadUrl, installationPath, (readBytes, totalBytes) => {
                const newPercentage = (readBytes / totalBytes) * 100;
                progress.report({
                    message: newPercentage.toFixed(0) + "%",
                    increment: newPercentage - lastPrecentage
                });

                lastPrecentage = newPercentage;
            });
        }
    );

    await fs.chmod(installationPath, 111); // Set xxx permissions
}
export async function ensureLanguageServerBinary(
    langServerSource: null | BinarySource
): Promise<null | string> {

    if (!langServerSource) {
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

    switch (langServerSource.type) {
        case BinarySourceType.ExplicitPath: {
            if (isBinaryAvailable(langServerSource.path)) {
                return langServerSource.path;
            }
            vscode.window.showErrorMessage(
                `Unable to execute ${'`'}${langServerSource.path} --version${'`'}. ` +
                "To use the bundled language server, set `rust-analyzer.raLspServerPath` " +
                "value to `null` or remove it from the settings to use it by default."
            );
            return null;
        }
        case BinarySourceType.GithubBinary: {
            const bundledBinaryPath = path.join(langServerSource.dir, langServerSource.file);

            if (!isBinaryAvailable(bundledBinaryPath)) {
                const userResponse = await vscode.window.showInformationMessage(
                    `Language server binary for rust-analyzer was not found. ` +
                    `Do you want to download it now?`,
                    "Download now", "Cancel"
                );
                if (userResponse !== "Download now") return null;

                try {
                    await downloadLatestLanguageServer(langServerSource);
                } catch (err) {
                    await vscode.window.showErrorMessage(
                        `Failed to download language server from ${langServerSource.repo.name} ` +
                        `GitHub repository: ${err.message}`
                    );
                    return null;
                }


                assert(
                    isBinaryAvailable(bundledBinaryPath),
                    "Downloaded language server binary is not functional"
                );

                vscode.window.showInformationMessage(
                    "Rust analyzer language server was successfully installed"
                );
            }
            return bundledBinaryPath;
        }
    }

    function isBinaryAvailable(binaryPath: string) {
        return spawnSync(binaryPath, ["--version"]).status === 0;
    }
}
