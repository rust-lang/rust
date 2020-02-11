import fetch from "node-fetch";
import * as fs from "fs";
import { strict as assert } from "assert";
import { NestedError } from "ts-nested-error";

class DownloadFileError extends NestedError {}

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
    const res = await fetch(url).catch(DownloadFileError.rethrow("Failed at initial fetch"));

    if (!res.ok) {
        console.log("Error", res.status, "while downloading file from", url);
        console.dir({ body: await res.text(), headers: res.headers }, { depth: 3 });

        throw new DownloadFileError(`Got response ${res.status}`);
    }

    const totalBytes = Number(res.headers.get('content-length'));
    assert(!Number.isNaN(totalBytes), "Sanity check of content-length protocol");

    let readBytes = 0;

    console.log("Downloading file of", totalBytes, "bytes size from", url, "to", destFilePath);

    // Here reject() may be called 2 times. As per ECMAScript standard, 2-d call is ignored
    // https://tc39.es/ecma262/#sec-promise-reject-functions

    return new Promise<void>((resolve, reject) => res.body
        .on("data", (chunk: Buffer) => {
            readBytes += chunk.length;
            onProgress(readBytes, totalBytes);
        })
        .on("error", err => reject(
            new DownloadFileError(`Read-stream error, read bytes: ${readBytes}`, err)
        ))
        .pipe(fs.createWriteStream(destFilePath, { mode: destFilePermissions }))
        .on("error", err => reject(
            new DownloadFileError(`Write-stream error, read bytes: ${readBytes}`, err)
        ))
        .on("close", resolve)
    );
}
