import fetch from "node-fetch";
import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";
import * as stream from "stream";
import * as util from "util";
import { log, assert } from "../util";
import { ArtifactReleaseInfo } from "./interfaces";

const pipeline = util.promisify(stream.pipeline);

/**
 * Downloads file from `url` and stores it at `destFilePath` with `destFilePermissions`.
 * `onProgress` callback is called on recieveing each chunk of bytes
 * to track the progress of downloading, it gets the already read and total
 * amount of bytes to read as its parameters.
 */
export async function downloadFile(
    url: string,
    destFilePath: fs.PathLike,
    destFilePermissions: number,
    onProgress: (readBytes: number, totalBytes: number) => void
): Promise<void> {
    const res = await fetch(url);

    if (!res.ok) {
        log.error("Error", res.status, "while downloading file from", url);
        log.error({ body: await res.text(), headers: res.headers });

        throw new Error(`Got response ${res.status} when trying to download a file.`);
    }

    const totalBytes = Number(res.headers.get('content-length'));
    assert(!Number.isNaN(totalBytes), "Sanity check of content-length protocol");

    log.debug("Downloading file of", totalBytes, "bytes size from", url, "to", destFilePath);

    let readBytes = 0;
    res.body.on("data", (chunk: Buffer) => {
        readBytes += chunk.length;
        onProgress(readBytes, totalBytes);
    });

    const destFileStream = fs.createWriteStream(destFilePath, { mode: destFilePermissions });

    await pipeline(res.body, destFileStream);
    return new Promise<void>(resolve => {
        destFileStream.on("close", resolve);
        destFileStream.destroy();

        // Details on workaround: https://github.com/rust-analyzer/rust-analyzer/pull/3092#discussion_r378191131
        // Issue at nodejs repo: https://github.com/nodejs/node/issues/31776
    });
}

/**
 * Downloads artifact from given `downloadUrl`.
 * Creates `installationDir` if it is not yet created and puts the artifact under
 * `artifactFileName`.
 * Displays info about the download progress in an info message printing the name
 * of the artifact as `displayName`.
 */
export async function downloadArtifactWithProgressUi(
    { downloadUrl, releaseName }: ArtifactReleaseInfo,
    artifactFileName: string,
    installationDir: string,
    displayName: string,
) {
    await fs.promises.mkdir(installationDir).catch(err => assert(
        err?.code === "EEXIST",
        `Couldn't create directory "${installationDir}" to download ` +
        `${artifactFileName} artifact: ${err?.message}`
    ));

    const installationPath = path.join(installationDir, artifactFileName);

    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            cancellable: false, // FIXME: add support for canceling download?
            title: `Downloading rust-analyzer ${displayName} (${releaseName})`
        },
        async (progress, _cancellationToken) => {
            let lastPrecentage = 0;
            const filePermissions = 0o755; // (rwx, r_x, r_x)
            await downloadFile(downloadUrl, installationPath, filePermissions, (readBytes, totalBytes) => {
                const newPercentage = (readBytes / totalBytes) * 100;
                progress.report({
                    message: newPercentage.toFixed(0) + "%",
                    increment: newPercentage - lastPrecentage
                });

                lastPrecentage = newPercentage;
            });
        }
    );
}
