// This package needs to be install:
//
// ```
// npm install browser-ui-test
// ```

const fs = require("fs");
const path = require("path");
const os = require('os');
const {Options, runTest} = require('browser-ui-test');

// If a test fails or errors, we will retry it two more times in case it was a flaky failure.
const NB_RETRY = 3;

function showHelp() {
    console.log("rustdoc-js options:");
    console.log("  --doc-folder [PATH]        : location of the generated doc folder");
    console.log("  --file [PATH]              : file to run (can be repeated)");
    console.log("  --debug                    : show extra information about script run");
    console.log("  --show-text                : render font in pages");
    console.log("  --no-headless              : disable headless mode");
    console.log("  --no-sandbox               : disable sandbox mode");
    console.log("  --help                     : show this message then quit");
    console.log("  --tests-folder [PATH]      : location of the .GOML tests folder");
    console.log("  --jobs [NUMBER]            : number of threads to run tests on");
    console.log("  --executable-path [PATH]   : path of the browser's executable to be used");
}

function isNumeric(s) {
    return /^\d+$/.test(s);
}

function parseOptions(args) {
    const opts = {
        "doc_folder": "",
        "tests_folder": "",
        "files": [],
        "debug": false,
        "show_text": false,
        "no_headless": false,
        "jobs": -1,
        "executable_path": null,
        "no_sandbox": false,
    };
    const correspondances = {
        "--doc-folder": "doc_folder",
        "--tests-folder": "tests_folder",
        "--debug": "debug",
        "--show-text": "show_text",
        "--no-headless": "no_headless",
        "--executable-path": "executable_path",
        "--no-sandbox": "no_sandbox",
    };

    for (let i = 0; i < args.length; ++i) {
        const arg = args[i];
        if (arg === "--doc-folder"
            || arg === "--tests-folder"
            || arg === "--file"
            || arg === "--jobs"
            || arg === "--executable-path") {
            i += 1;
            if (i >= args.length) {
                console.log("Missing argument after `" + arg + "` option.");
                return null;
            }
            const arg_value = args[i];
            if (arg === "--jobs") {
                if (!isNumeric(arg_value)) {
                    console.log(
                        "`--jobs` option expects a positive number, found `" + arg_value + "`");
                    return null;
                }
                opts["jobs"] = parseInt(arg_value);
            } else if (arg !== "--file") {
                opts[correspondances[arg]] = arg_value;
            } else {
                opts["files"].push(arg_value);
            }
        } else if (arg === "--help") {
            showHelp();
            process.exit(0);
        } else if (arg === "--no-sandbox") {
            console.log("`--no-sandbox` is being used. Be very careful!");
            opts[correspondances[arg]] = true;
        } else if (correspondances[arg]) {
            opts[correspondances[arg]] = true;
        } else {
            console.log("Unknown option `" + arg + "`.");
            console.log("Use `--help` to see the list of options");
            return null;
        }
    }
    if (opts["tests_folder"].length < 1) {
        console.log("Missing `--tests-folder` option.");
    } else if (opts["doc_folder"].length < 1) {
        console.log("Missing `--doc-folder` option.");
    } else {
        return opts;
    }
    return null;
}

/// Print single char status information without \n
function char_printer(n_tests) {
    const max_per_line = 10;
    let current = 0;
    return {
        successful: function() {
            current += 1;
            if (current % max_per_line === 0) {
                process.stdout.write(`. (${current}/${n_tests})${os.EOL}`);
            } else {
                process.stdout.write(".");
            }
        },
        erroneous: function() {
            current += 1;
            if (current % max_per_line === 0) {
                process.stderr.write(`F (${current}/${n_tests})${os.EOL}`);
            } else {
                process.stderr.write("F");
            }
        },
        finish: function() {
            if (current % max_per_line === 0) {
                // Don't output if we are already at a matching line end
                console.log("");
            } else {
                const spaces = " ".repeat(max_per_line - (current % max_per_line));
                process.stdout.write(`${spaces} (${current}/${n_tests})${os.EOL}${os.EOL}`);
            }
        },
    };
}

// Sort array by .file_name property
function by_filename(a, b) {
    return a.file_name - b.file_name;
}

async function runTests(opts, framework_options, files, results, status_bar, showTestFailures) {
    const tests_queue = [];

    for (const testPath of files) {
        const callback = runTest(testPath, framework_options)
            .then(out => {
                const [output, nb_failures] = out;
                results[nb_failures === 0 ? "successful" : "failed"].push({
                    file_name: testPath,
                    output: output,
                });
                if (nb_failures === 0) {
                    status_bar.successful();
                } else if (showTestFailures) {
                    status_bar.erroneous();
                }
            })
            .catch(err => {
                results.errored.push({
                    file_name: testPath,
                    output: err,
                });
                if (showTestFailures) {
                    status_bar.erroneous();
                }
            })
            .finally(() => {
                // We now remove the promise from the tests_queue.
                tests_queue.splice(tests_queue.indexOf(callback), 1);
            });
        tests_queue.push(callback);
        if (opts["jobs"] > 0 && tests_queue.length >= opts["jobs"]) {
            await Promise.race(tests_queue);
        }
    }
    if (tests_queue.length > 0) {
        await Promise.all(tests_queue);
    }
}

function createEmptyResults() {
    return {
        successful: [],
        failed: [],
        errored: [],
    };
}

async function main(argv) {
    const opts = parseOptions(argv.slice(2));
    if (opts === null) {
        process.exit(1);
    }

    // Print successful tests too
    let debug = false;
    // Run tests in sequentially
    let headless = true;
    const framework_options = new Options();
    try {
        // This is more convenient that setting fields one by one.
        let args = [
            "--variable", "DOC_PATH", opts["doc_folder"], "--enable-fail-on-js-error",
            "--allow-file-access-from-files",
        ];
        if (opts["debug"]) {
            debug = true;
            args.push("--debug");
        }
        if (opts["show_text"]) {
            args.push("--show-text");
        }
        if (opts["no_sandbox"]) {
            args.push("--no-sandbox");
        }
        if (opts["no_headless"]) {
            args.push("--no-headless");
            headless = false;
        }
        if (opts["executable_path"] !== null) {
            args.push("--executable-path");
            args.push(opts["executable_path"]);
        }
        framework_options.parseArguments(args);
    } catch (error) {
        console.error(`invalid argument: ${error}`);
        process.exit(1);
    }

    let files;
    if (opts["files"].length === 0) {
        files = fs.readdirSync(opts["tests_folder"]);
    } else {
        files = opts["files"];
    }
    files = files.filter(file => path.extname(file) == ".goml");
    if (files.length === 0) {
        console.error("rustdoc-gui: No test selected");
        process.exit(2);
    }
    files.forEach((file_name, index) => {
        files[index] = path.join(opts["tests_folder"], file_name);
    });
    files.sort();

    if (!headless) {
        opts["jobs"] = 1;
        console.log("`--no-headless` option is active, disabling concurrency for running tests.");
    }

    console.log(`Running ${files.length} rustdoc-gui (${opts["jobs"]} concurrently) ...`);

    if (opts["jobs"] < 1) {
        process.setMaxListeners(files.length + 1);
    } else if (headless) {
        process.setMaxListeners(opts["jobs"] + 1);
    }

    // We catch this "event" to display a nicer message in case of unexpected exit (because of a
    // missing `--no-sandbox`).
    const exitHandling = (code) => {
        if (!opts["no_sandbox"]) {
            console.log("");
            console.log(
                "`browser-ui-test` crashed unexpectedly. Please try again with adding `--test-args \
--no-sandbox` at the end. For example: `x.py test tests/rustdoc-gui --test-args --no-sandbox`");
            console.log("");
        }
    };
    process.on('exit', exitHandling);

    const originalFilesLen = files.length;
    let results = createEmptyResults();
    const status_bar = char_printer(files.length);

    let new_results;
    for (let it = 0; it < NB_RETRY && files.length > 0; ++it) {
        new_results = createEmptyResults();
        await runTests(opts, framework_options, files, new_results, status_bar, it + 1 >= NB_RETRY);
        Array.prototype.push.apply(results.successful, new_results.successful);
        // We generate the new list of files with the previously failing tests.
        files = Array.prototype.concat(new_results.failed, new_results.errored).map(
            f => f['file_name']);
        if (files.length > originalFilesLen / 2) {
            // If we have too many failing tests, it's very likely not flaky failures anymore so
            // no need to retry.
            break;
        }
    }

    status_bar.finish();

    Array.prototype.push.apply(results.failed, new_results.failed);
    Array.prototype.push.apply(results.errored, new_results.errored);

    // We don't need this listener anymore.
    process.removeListener("exit", exitHandling);

    if (debug) {
        results.successful.sort(by_filename);
        results.successful.forEach(r => {
            console.log(r.output);
        });
    }

    if (results.failed.length > 0) {
        console.log("");
        results.failed.sort(by_filename);
        results.failed.forEach(r => {
            console.log(r.file_name, r.output);
        });
    }
    if (results.errored.length > 0) {
        console.log(os.EOL);
        // print run errors on the bottom so developers see them better
        results.errored.sort(by_filename);
        results.errored.forEach(r => {
            console.error(r.file_name, r.output);
        });
    }

    if (results.failed.length > 0 || results.errored.length > 0) {
        process.exit(1);
    }
}

main(process.argv);
