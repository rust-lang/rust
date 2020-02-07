import fetch from "node-fetch";
import { throttle } from "throttle-debounce";
import * as fs from "fs";

export async function downloadFile(
    url: string,
    destFilePath: fs.PathLike,
    onProgress: (readBytes: number, totalBytes: number) => void
): Promise<void> {
    onProgress = throttle(100, /* noTrailing: */ true, onProgress);

    const response = await fetch(url);

    const totalBytes = Number(response.headers.get('content-length'));
    let readBytes = 0;

    return new Promise<void>((resolve, reject) => response.body
        .on("data", (chunk: Buffer) => {
            readBytes += chunk.length;
            onProgress(readBytes, totalBytes);
        })
        .on("end", resolve)
        .on("error", reject)
        .pipe(fs.createWriteStream(destFilePath))
    );
}
