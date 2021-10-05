// This package needs to be install:
//
// ```
// npm install browser-ui-test
// ```

const fs = require("fs");
const path = require("path");
const os = require('os');
const {Options, runTest} = require('browser-ui-test');

function showHelp() {
    console.log("rustdoc-js options:");
    console.log("  --doc-folder [PATH]        : location of the generated doc folder");
    console.log("  --file [PATH]              : file to run (can be repeated)");
    console.log("  --debug                    : show extra information about script run");
    console.log("  --show-text                : render font in pages");
    console.log("  --no-headless              : disable headless mode");
    console.log("  --help                     : show this message then quit");
    console.log("  --tests-folder [PATH]      : location of the .GOML tests folder");
    console.log("  --jobs [NUMBER]            : number of threads to run tests on");
}

function isNumeric(s) {
    return /^\d+$/.test(s);
}

function parseOptions(args) {
    var opts = {
        "doc_folder": "",
        "tests_folder": "",
        "files": [],
        "debug": false,
        "show_text": false,
        "no_headless": false,
        "jobs": -1,
    };
    var correspondances = {
        "--doc-folder": "doc_folder",
        "--tests-folder": "tests_folder",
        "--debug": "debug",
        "--show-text": "show_text",
        "--no-headless": "no_headless",
    };

    for (var i = 0; i < args.length; ++i) {
        if (args[i] === "--doc-folder"
            || args[i] === "--tests-folder"
            || args[i] === "--file"
            || args[i] === "--jobs") {
            i += 1;
            if (i >= args.length) {
                console.log("Missing argument after `" + args[i - 1] + "` option.");
                return null;
            }
            if (args[i - 1] === "--jobs") {
                if (!isNumeric(args[i])) {
                    console.log(
                        "`--jobs` option expects a positive number, found `" + args[i] + "`");
                    return null;
                }
                opts["jobs"] = parseInt(args[i]);
            } else if (args[i - 1] !== "--file") {
                opts[correspondances[args[i - 1]]] = args[i];
            } else {
                opts["files"].push(args[i]);
            }
        } else if (args[i] === "--help") {
            showHelp();
            process.exit(0);
        } else if (correspondances[args[i]]) {
            opts[correspondances[args[i]]] = true;
        } else {
            console.log("Unknown option `" + args[i] + "`.");
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

/// Sort array by .file_name property
function by_filename(a, b) {
    return a.file_name - b.file_name;
}

async function main(argv) {
    let opts = parseOptions(argv.slice(2));
    if (opts === null) {
        process.exit(1);
    }

    // Print successful tests too
    let debug = false;
    // Run tests in sequentially
    let no_headless = false;
    const options = new Options();
    try {
        // This is more convenient that setting fields one by one.
        let args = [
            "--no-screenshot",
            "--variable", "DOC_PATH", opts["doc_folder"],
        ];
        if (opts["debug"]) {
            debug = true;
            args.push("--debug");
        }
        if (opts["show_text"]) {
            args.push("--show-text");
        }
        if (opts["no_headless"]) {
            args.push("--no-headless");
            no_headless = true;
        }
        options.parseArguments(args);
    } catch (error) {
        console.error(`invalid argument: ${error}`);
        process.exit(1);
    }

    let failed = false;
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
    files.sort();

    console.log(`Running ${files.length} rustdoc-gui (${opts["jobs"]} concurrently) ...`);

    if (opts["jobs"] < 1) {
        process.setMaxListeners(files.length + 1);
    } else {
        process.setMaxListeners(opts["jobs"] + 1);
    }

    const tests_queue = [];
    let results = {
        successful: [],
        failed: [],
        errored: [],
    };
    const status_bar = char_printer(files.length);
    for (let i = 0; i < files.length; ++i) {
        const file_name = files[i];
        const testPath = path.join(opts["tests_folder"], file_name);
        const callback = runTest(testPath, options)
            .then(out => {
                const [output, nb_failures] = out;
                results[nb_failures === 0 ? "successful" : "failed"].push({
                    file_name: file_name,
                    output: output,
                });
                if (nb_failures > 0) {
                    status_bar.erroneous();
                    failed = true;
                } else {
                    status_bar.successful();
                }
            })
            .catch(err => {
                results.errored.push({
                    file_name: file_name,
                    output: err,
                });
                status_bar.erroneous();
                failed = true;
            })
            .finally(() => {
                // We now remove the promise from the tests_queue.
                tests_queue.splice(tests_queue.indexOf(callback), 1);
            });
        tests_queue.push(callback);
        if (no_headless) {
            await tests_queue[i];
        } else if (opts["jobs"] > 0 && tests_queue.length >= opts["jobs"]) {
            await Promise.race(tests_queue);
        }
    }
    if (!no_headless && tests_queue.length > 0) {
        await Promise.all(tests_queue);
    }
    status_bar.finish();

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
            console.log(r.output);
        });
    }
    if (results.errored.length > 0) {
        console.log(os.EOL);
        // print run errors on the bottom so developers see them better
        results.errored.sort(by_filename);
        results.errored.forEach(r => {
            console.error(r.output);
        });
    }

    if (failed) {
        process.exit(1);
    }
}

main(process.argv);
