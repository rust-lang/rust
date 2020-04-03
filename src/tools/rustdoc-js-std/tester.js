const fs = require('fs');
const path = require('path');
const tools = require('../rustdoc-js-common/lib.js');


function findFile(dir, name, extension) {
    var entries = fs.readdirSync(dir);
    var matches = [];
    for (var i = 0; i < entries.length; ++i) {
        var entry = entries[i];
        var file_type = fs.statSync(dir + entry);
        if (file_type.isDirectory()) {
            continue;
        }
        if (entry.startsWith(name) && entry.endsWith(extension)) {
            var version = entry.slice(name.length, entry.length - extension.length);
            version = version.split(".").map(function(x) {
                return parseInt(x);
            });
            var total = 0;
            var mult = 1;
            for (var j = version.length - 1; j >= 0; --j) {
                total += version[j] * mult;
                mult *= 1000;
            }
            matches.push([entry, total]);
        }
    }
    if (matches.length === 0) {
        return null;
    }
    // We make a reverse sort to have the "highest" file. Very useful in case you didn't clean up
    // you std doc folder...
    matches.sort(function(a, b) {
        return b[1] - a[1];
    });
    return matches[0][0];
}

function readFileMatching(dir, name, extension) {
    if (dir.endsWith("/") === false) {
        dir += "/";
    }
    var f = findFile(dir, name, extension);
    if (f === null) {
        return "";
    }
    return tools.readFile(dir + f);
}

function main(argv) {
    if (argv.length !== 4) {
        console.error("USAGE: node tester.js STD_DOCS TEST_FOLDER");
        return 1;
    }
    var std_docs = argv[2];
    var test_folder = argv[3];

    var mainJs = readFileMatching(std_docs, "main", ".js");
    var aliases = readFileMatching(std_docs, "aliases", ".js");
    var searchIndex = readFileMatching(std_docs, "search-index", ".js").split("\n");

    var [loaded, index] = tools.loadMainJsAndIndex(mainJs, aliases, searchIndex, "std");

    var errors = 0;

    fs.readdirSync(test_folder).forEach(function(file) {
        process.stdout.write('Checking "' + file + '" ... ');
        errors += tools.runChecks(path.join(test_folder, file), loaded, index);
    });
    return errors > 0 ? 1 : 0;
}

process.exit(main(process.argv));
