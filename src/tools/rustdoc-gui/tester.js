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
    console.log("  --help                     : show this message then quit");
    console.log("  --tests-folder [PATH]      : location of the .GOML tests folder");
}

function parseOptions(args) {
    var opts = {
        "doc_folder": "",
        "tests_folder": "",
    };
    var correspondances = {
        "--doc-folder": "doc_folder",
        "--tests-folder": "tests_folder",
    };

    for (var i = 0; i < args.length; ++i) {
        if (args[i] === "--doc-folder"
            || args[i] === "--tests-folder") {
            i += 1;
            if (i >= args.length) {
                console.log("Missing argument after `" + args[i - 1] + "` option.");
                return null;
            }
            opts[correspondances[args[i - 1]]] = args[i];
        } else if (args[i] === "--help") {
            showHelp();
            process.exit(0);
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
        options.parseArguments([
            "--no-screenshot",
            "--variable", "DOC_PATH", opts["doc_folder"],
        ]);
    } catch (error) {
        console.error(`invalid argument: ${error}`);
        process.exit(1);
    }

    let failed = false;
    let files = fs.readdirSync(opts["tests_folder"]).filter(file => path.extname(file) == ".goml");

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
