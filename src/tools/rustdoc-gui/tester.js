// This package needs to be install:
//
// ```
// npm install browser-ui-test
// ```
const path = require('path');
const {Options, runTest} = require('browser-ui-test');

function showHelp() {
    console.log("rustdoc-js options:");
    console.log("  --doc-folder [PATH]        : location of the generated doc folder");
    console.log("  --help                     : show this message then quit");
    console.log("  --test-file [PATH]         : location of the JS test file");
}

function parseOptions(args) {
    var opts = {
        "doc_folder": "",
        "test_file": "",
    };
    var correspondances = {
        "--doc-folder": "doc_folder",
        "--test-file": "test_file",
    };

    for (var i = 0; i < args.length; ++i) {
        if (args[i] === "--doc-folder"
            || args[i] === "--test-file") {
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
    if (opts["test_file"].length < 1) {
        console.log("Missing `--test-file` option.");
    } else if (opts["doc_folder"].length < 1) {
        console.log("Missing `--doc-folder` option.");
    } else {
        return opts;
    }
    return null;
}

function checkFile(test_file, opts, loaded, index) {
    const test_name = path.basename(test_file, ".js");

    process.stdout.write('Checking "' + test_name + '" ... ');
    return runChecks(test_file, loaded, index);
}

function main(argv) {
    var opts = parseOptions(argv.slice(2));
    if (opts === null) {
        process.exit(1);
    }

    const options = new Options();
    try {
        // This is more convenient that setting fields one by one.
        options.parseArguments([
            '--no-screenshot',
            "--variable", "DOC_PATH", opts["doc_folder"],
        ]);
    } catch (error) {
        console.error(`invalid argument: ${error}`);
        process.exit(1);
    }

    runTest(opts["test_file"], options).then(out => {
        const [output, nb_failures] = out;
        console.log(output);
        process.exit(nb_failures);
    }).catch(err => {
        console.error(err);
        process.exit(1);
    });
}

main(process.argv);
