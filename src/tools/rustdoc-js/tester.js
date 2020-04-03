const fs = require('fs');
const path = require('path');
const tools = require('../rustdoc-js-common/lib.js');

function load_files(out_folder, crate) {
    var mainJs = tools.readFile(out_folder + "/main.js");
    var aliases = tools.readFile(out_folder + "/aliases.js");
    var searchIndex = tools.readFile(out_folder + "/search-index.js").split("\n");

    return tools.loadMainJsAndIndex(mainJs, aliases, searchIndex, crate);
}

function main(argv) {
    if (argv.length < 4) {
        console.error("USAGE: node tester.js OUT_FOLDER [TESTS]");
        return 1;
    }
    if (argv[2].substr(-1) !== "/") {
        argv[2] += "/";
    }
    const out_folder = argv[2];

    var errors = 0;

    for (var j = 3; j < argv.length; ++j) {
        const test_file = argv[j];
        const test_name = path.basename(test_file, ".js");

        process.stdout.write('Checking "' + test_name + '" ... ');
        if (!fs.existsSync(test_file)) {
            errors += 1;
            console.error("FAILED");
            console.error("==> Missing '" + test_name + ".js' file...");
            continue;
        }

        const test_out_folder = out_folder + test_name;

        var [loaded, index] = load_files(test_out_folder, test_name);
        errors += tools.runChecks(test_file, loaded, index);
    }
    return errors > 0 ? 1 : 0;
}

process.exit(main(process.argv));
