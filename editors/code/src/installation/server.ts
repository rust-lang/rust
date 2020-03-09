import * as vscode from "vscode";
import * as path from "path";
import { spawnSync } from "child_process";

import { ArtifactSource } from "./interfaces";
import { fetchArtifactReleaseInfo } from "./fetch_artifact_release_info";
import { downloadArtifactWithProgressUi } from "./downloads";
import { log, assert } from "../util";
import { Config, NIGHTLY_TAG } from "../config";

export async function ensureServerBinary(config: Config): Promise<null | string> {
    const source = config.serverSource;

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
        case ArtifactSource.Type.ExplicitPath: {
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
        case ArtifactSource.Type.GithubRelease: {
            if (!shouldDownloadServer(source, config)) {
                return path.join(source.dir, source.file);
            }

            if (config.askBeforeDownload) {
                const userResponse = await vscode.window.showInformationMessage(
                    `Language server version ${source.tag} for rust-analyzer is not installed. ` +
                    "Do you want to download it now?",
                    "Download now", "Cancel"
                );
                if (userResponse !== "Download now") return null;
            }

            return await downloadServer(source, config);
        }
    }
}

function shouldDownloadServer(
    source: ArtifactSource.GithubRelease,
    config: Config
): boolean {
    if (!isBinaryAvailable(path.join(source.dir, source.file))) return true;

    const installed = {
        tag: config.serverReleaseTag.get(),
        date: config.serverReleaseDate.get()
    };
    const required = {
        tag: source.tag,
        date: config.installedNightlyExtensionReleaseDate.get()
    };

    log.debug("Installed server:", installed, "required:", required);

    if (required.tag !== NIGHTLY_TAG || installed.tag !== NIGHTLY_TAG) {
        return required.tag !== installed.tag;
    }

    assert(required.date !== null, "Extension release date should have been saved during its installation");
    assert(installed.date !== null, "Server release date should have been saved during its installation");

    return installed.date.getTime() !== required.date.getTime();
}

async function downloadServer(
    source: ArtifactSource.GithubRelease,
    config: Config,
): Promise<null | string> {
    try {
        const releaseInfo = await fetchArtifactReleaseInfo(source.repo, source.file, source.tag);

        await downloadArtifactWithProgressUi(releaseInfo, source.file, source.dir, "language server");
        await Promise.all([
            config.serverReleaseTag.set(releaseInfo.releaseName),
            config.serverReleaseDate.set(releaseInfo.releaseDate)
        ]);
    } catch (err) {
        log.downloadError(err, "language server", source.repo.name);
        return null;
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

    return binaryPath;
}

function isBinaryAvailable(binaryPath: string): boolean {
    const res = spawnSync(binaryPath, ["--version"]);

    // ACHTUNG! `res` type declaration is inherently wrong, see
    // https://github.com/DefinitelyTyped/DefinitelyTyped/issues/42221

    log.debug("Checked binary availablity via --version", res);
    log.debug(binaryPath, "--version output:", res.output?.map(String));

    return res.status === 0;
}
