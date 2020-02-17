import * as vscode from "vscode";
import * as path from "path";
import { promises as fs } from "fs";
import { strict as assert } from "assert";

import { ArtifactReleaseInfo } from "./interfaces";
import { downloadFile } from "./download_file";
import { throttle } from "throttle-debounce";

/**
 * Downloads artifact from given `downloadUrl`.
 * Creates `installationDir` if it is not yet created and put the artifact under
 * `artifactFileName`.
 * Displays info about the download progress in an info message printing the name
 * of the artifact as `displayName`.
 */
export async function downloadArtifact(
    { downloadUrl, releaseName }: ArtifactReleaseInfo,
    artifactFileName: string,
    installationDir: string,
    displayName: string,
) {
    await fs.mkdir(installationDir).catch(err => assert.strictEqual(
        err?.code,
        "EEXIST",
        `Couldn't create directory "${installationDir}" to download ` +
        `${artifactFileName} artifact: ${err.message}`
    ));

    const installationPath = path.join(installationDir, artifactFileName);

    console.time(`Downloading ${artifactFileName}`);
    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            cancellable: false, // FIXME: add support for canceling download?
            title: `Downloading ${displayName} (${releaseName})`
        },
        async (progress, _cancellationToken) => {
            let lastPrecentage = 0;
            const filePermissions = 0o755; // (rwx, r_x, r_x)
            await downloadFile(downloadUrl, installationPath, filePermissions, throttle(
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
    console.timeEnd(`Downloading ${artifactFileName}`);
}
