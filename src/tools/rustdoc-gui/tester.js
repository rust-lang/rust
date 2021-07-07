// This package needs to be install:
//
// ```
// npm install browser-ui-test
// ```
const fs = require("fs");
const path = require("path");
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
}

function parseOptions(args) {
    var opts = {
        "doc_folder": "",
        "tests_folder": "",
        "files": [],
        "debug": false,
        "show_text": false,
        "no_headless": false,
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
            || args[i] === "--file") {
            i += 1;
            if (i >= args.length) {
                console.log("Missing argument after `" + args[i - 1] + "` option.");
                return null;
            }
            if (args[i - 1] !== "--file") {
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

async function main(argv) {
    let opts = parseOptions(argv.slice(2));
    if (opts === null) {
        process.exit(1);
    }

    const options = new Options();
    try {
        // This is more convenient that setting fields one by one.
        let args = [
            "--no-screenshot",
            "--variable", "DOC_PATH", opts["doc_folder"],
        ];
        if (opts["debug"]) {
            args.push("--debug");
        }
        if (opts["show_text"]) {
            args.push("--show-text");
        }
        if (opts["no_headless"]) {
            args.push("--no-headless");
        }
        options.parseArguments(args);
    } catch (error) {
        console.error(`invalid argument: ${error}`);
        process.exit(1);
    }

    let failed = false;
    let files;
    if (opts["files"].length === 0) {
        files = fs.readdirSync(opts["tests_folder"]).filter(file => path.extname(file) == ".goml");
    } else {
        files = opts["files"].filter(file => path.extname(file) == ".goml");
    }

    files.sort();
    for (var i = 0; i < files.length; ++i) {
        const testPath = path.join(opts["tests_folder"], files[i]);
        await runTest(testPath, options).then(out => {
            const [output, nb_failures] = out;
            console.log(output);
            if (nb_failures > 0) {
                failed = true;
            }
        }).catch(err => {
            console.error(err);
            failed = true;
        });
    }
    if (failed) {
        process.exit(1);
    }
}

main(process.argv);
