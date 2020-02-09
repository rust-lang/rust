import * as vscode from "vscode";
import * as path from "path";
import { strict as assert } from "assert";
import { promises as fs } from "fs";
import { promises as dns } from "dns";
import { spawnSync } from "child_process";
import { throttle } from "throttle-debounce";

import { BinarySource } from "./interfaces";
import { fetchLatestArtifactMetadata } from "./fetch_latest_artifact_metadata";
import { downloadFile } from "./download_file";

export async function downloadLatestLanguageServer(
    {file: artifactFileName, dir: installationDir, repo}: BinarySource.GithubRelease
) {
    const { releaseName, downloadUrl } = (await fetchLatestArtifactMetadata(
        repo, artifactFileName
    ))!;

    await fs.mkdir(installationDir).catch(err => assert.strictEqual(
        err?.code,
        "EEXIST",
        `Couldn't create directory "${installationDir}" to download `+
        `language server binary: ${err.message}`
    ));

    const installationPath = path.join(installationDir, artifactFileName);

    console.time("Downloading ra_lsp_server");
    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            cancellable: false, // FIXME: add support for canceling download?
            title: `Downloading language server (${releaseName})`
        },
        async (progress, _cancellationToken) => {
            let lastPrecentage = 0;
            await downloadFile(downloadUrl, installationPath, throttle(
                200,
                /* noTrailing: */ true,
                (readBytes, totalBytes) => {
                    const newPercentage = (readBytes / totalBytes) * 100;
                    progress.report({
                        message: newPercentage.toFixed(0) + "%",
                        increment: newPercentage - lastPrecentage
                    });

                    lastPrecentage = newPercentage;
                })
            );
        }
    );
    console.timeEnd("Downloading ra_lsp_server");

    await fs.chmod(installationPath, 0o755); // Set (rwx, r_x, r_x) permissions
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
        case BinarySource.Type.ExplicitPath: {
            if (isBinaryAvailable(langServerSource.path)) {
                return langServerSource.path;
            }

            vscode.window.showErrorMessage(
                `Unable to run ${langServerSource.path} binary. ` +
                `To use the pre-built language server, set "rust-analyzer.raLspServerPath" ` +
                "value to `null` or remove it from the settings to use it by default."
            );
            return null;
        }
        case BinarySource.Type.GithubRelease: {
            const prebuiltBinaryPath = path.join(langServerSource.dir, langServerSource.file);

            if (isBinaryAvailable(prebuiltBinaryPath)) {
                return prebuiltBinaryPath;
            }

            const userResponse = await vscode.window.showInformationMessage(
                "Language server binary for rust-analyzer was not found. " +
                "Do you want to download it now?",
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

                await dns.resolve('www.google.com').catch(err => {
                    console.error("DNS resolution failed, there might be an issue with Internet availability");
                    console.error(err);
                });

                return null;
            }

            if (!isBinaryAvailable(prebuiltBinaryPath)) assert(false,
                `Downloaded language server binary is not functional.` +
                `Downloaded from: ${JSON.stringify(langServerSource)}`
            );


            vscode.window.showInformationMessage(
                "Rust analyzer language server was successfully installed 🦀"
            );

            return prebuiltBinaryPath;
        }
    }

    function isBinaryAvailable(binaryPath: string) {
        const res = spawnSync(binaryPath, ["--version"]);

        // ACHTUNG! `res` type declaration is inherently wrong, see
        // https://github.com/DefinitelyTyped/DefinitelyTyped/issues/42221

        console.log("Checked binary availablity via --version", res);
        console.log(binaryPath, "--version output:", res.output?.map(String));

        return res.status === 0;
    }
}
