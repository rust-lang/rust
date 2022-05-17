import * as path from "path";
import * as fs from "fs";

import { runTests } from "@vscode/test-electron";

async function main() {
    // The folder containing the Extension Manifest package.json
    // Passed to `--extensionDevelopmentPath`
    const extensionDevelopmentPath = path.resolve(__dirname, "../../");

    // Minimum supported version.
    const jsonData = fs.readFileSync(path.join(extensionDevelopmentPath, "package.json"));
    const json = JSON.parse(jsonData.toString());
    let minimalVersion: string = json.engines.vscode;
    if (minimalVersion.startsWith("^")) minimalVersion = minimalVersion.slice(1);

    const launchArgs = ["--disable-extensions", extensionDevelopmentPath];

    // All test suites (either unit tests or integration tests) should be in subfolders.
    const extensionTestsPath = path.resolve(__dirname, "./unit/index");

    // Run tests using the minimal supported version.
    await runTests({
        version: minimalVersion,
        launchArgs,
        extensionDevelopmentPath,
        extensionTestsPath,
    });

    // and the latest one
    await runTests({
        version: "stable",
        launchArgs,
        extensionDevelopmentPath,
        extensionTestsPath,
    });
}

main().catch((err) => {
    // eslint-disable-next-line no-console
    console.error("Failed to run tests", err);
    process.exit(1);
});
