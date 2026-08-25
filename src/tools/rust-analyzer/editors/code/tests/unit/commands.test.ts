import * as assert from "node:assert/strict";

import {
    cargoNewArgs,
    determineNewProjectOpenAction,
    validateNewProjectName,
} from "../../src/new_project";
import type { Context } from ".";

export async function getTests(ctx: Context) {
    await ctx.suite("New project command", (suite) => {
        suite.addTest("rejects empty project name", async () => {
            assert.equal(validateNewProjectName("", []), "Project name cannot be empty.");
            assert.equal(validateNewProjectName("   ", []), "Project name cannot be empty.");
        });

        suite.addTest("rejects dot project names", async () => {
            assert.equal(validateNewProjectName(".", []), "Project name cannot be '.' or '..'.");
            assert.equal(validateNewProjectName("..", []), "Project name cannot be '.' or '..'.");
        });

        suite.addTest("rejects path separators", async () => {
            assert.equal(
                validateNewProjectName("foo/bar", []),
                "Project name cannot contain '/' or '\\' characters.",
            );
            assert.equal(
                validateNewProjectName("foo\\bar", []),
                "Project name cannot contain '/' or '\\' characters.",
            );
        });

        suite.addTest("rejects invalid Cargo package name characters", async () => {
            assert.equal(
                validateNewProjectName("foo.bar", []),
                "Project name can contain only alphanumeric characters, '-' or '_'.",
            );
            assert.equal(
                validateNewProjectName("foo bar", []),
                "Project name can contain only alphanumeric characters, '-' or '_'.",
            );
            assert.equal(
                validateNewProjectName("foo+bar", []),
                "Project name can contain only alphanumeric characters, '-' or '_'.",
            );
        });

        suite.addTest("rejects existing child folder collisions", async () => {
            assert.equal(
                validateNewProjectName("demo", ["demo"]),
                "A file or folder with this name already exists.",
            );
        });

        suite.addTest("accepts a normal project name", async () => {
            assert.equal(validateNewProjectName("demo-project", []), undefined);
        });

        suite.addTest("resolves addToWorkspace fallback without workspace", async () => {
            assert.equal(determineNewProjectOpenAction("addToWorkspace", false), "open");
        });

        suite.addTest("keeps addToWorkspace when workspace exists", async () => {
            assert.equal(determineNewProjectOpenAction("addToWorkspace", true), "addToWorkspace");
        });

        suite.addTest("defaults to ask for unknown values", async () => {
            assert.equal(determineNewProjectOpenAction(undefined, true), "ask");
            assert.equal(determineNewProjectOpenAction("ask", true), "ask");
        });

        suite.addTest("builds binary cargo args", async () => {
            assert.deepEqual(cargoNewArgs("bin", "demo"), ["new", "--bin", "demo"]);
        });

        suite.addTest("builds library cargo args", async () => {
            assert.deepEqual(cargoNewArgs("lib", "demo"), ["new", "--lib", "demo"]);
        });
    });
}
