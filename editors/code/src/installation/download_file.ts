import fetch from "node-fetch";
import * as fs from "fs";
import { strict as assert } from "assert";

/**
 * Downloads file from `url` and stores it at `destFilePath`.
 * `onProgress` callback is called on recieveing each chunk of bytes
 * to track the progress of downloading, it gets the already read and total
 * amount of bytes to read as its parameters.
 */
export async function downloadFile(
    url: string,
    destFilePath: fs.PathLike,
    onProgress: (readBytes: number, totalBytes: number) => void
): Promise<void> {
    const res = await fetch(url);

    if (!res.ok) {
        console.log("Error", res.status, "while downloading file from", url);
        console.dir({ body: await res.text(), headers: res.headers }, { depth: 3 });

        throw new Error(`Got response ${res.status} when trying to download a file`);
    }

    const totalBytes = Number(res.headers.get('content-length'));
    assert(!Number.isNaN(totalBytes), "Sanity check of content-length protocol");

    let readBytes = 0;

    console.log("Downloading file of", totalBytes, "bytes size from", url, "to", destFilePath);

    return new Promise<void>((resolve, reject) => res.body
        .on("data", (chunk: Buffer) => {
            readBytes += chunk.length;
            onProgress(readBytes, totalBytes);
        })
        .on("end", resolve)
        .on("error", reject)
        .pipe(fs.createWriteStream(destFilePath))
    );
}
