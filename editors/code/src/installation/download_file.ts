import fetch from "node-fetch";
import * as fs from "fs";
import * as stream from "stream";
import * as util from "util";
import { strict as assert } from "assert";

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
        console.log("Error", res.status, "while downloading file from", url);
        console.dir({ body: await res.text(), headers: res.headers }, { depth: 3 });

        throw new Error(`Got response ${res.status} when trying to download a file.`);
    }

    const totalBytes = Number(res.headers.get('content-length'));
    assert(!Number.isNaN(totalBytes), "Sanity check of content-length protocol");

    console.log("Downloading file of", totalBytes, "bytes size from", url, "to", destFilePath);

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
