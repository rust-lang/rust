// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const fs = require('fs');

const TEST_FOLDER = 'src/test/rustdoc-js/';

// Stupid function extractor based on indent.
function extractFunction(content, functionName) {
    var x = content.split('\n');
    var in_func = false;
    var indent = 0;
    var lines = [];

    for (var i = 0; i < x.length; ++i) {
        if (in_func === false) {
            var splitter = "function " + functionName + "(";
            if (x[i].trim().startsWith(splitter)) {
                in_func = true;
                indent = x[i].split(splitter)[0].length;
                lines.push(x[i]);
            }
        } else {
            lines.push(x[i]);
            if (x[i].trim() === "}" && x[i].split("}")[0].length === indent) {
                return lines.join("\n");
            }
        }
    }
    return null;
}

// Stupid function extractor for array.
function extractArrayVariable(content, arrayName) {
    var x = content.split('\n');
    var found_var = false;
    var lines = [];

    for (var i = 0; i < x.length; ++i) {
        if (found_var === false) {
            var splitter = "var " + arrayName + " = [";
            if (x[i].trim().startsWith(splitter)) {
                found_var = true;
                i -= 1;
            }
        } else {
            lines.push(x[i]);
            if (x[i].endsWith('];')) {
                return lines.join("\n");
            }
        }
    }
    return null;
}

// Stupid function extractor for variable.
function extractVariable(content, varName) {
    var x = content.split('\n');
    var found_var = false;
    var lines = [];

    for (var i = 0; i < x.length; ++i) {
        if (found_var === false) {
            var splitter = "var " + varName + " = ";
            if (x[i].trim().startsWith(splitter)) {
                found_var = true;
                i -= 1;
            }
        } else {
            lines.push(x[i]);
            if (x[i].endsWith(';')) {
                return lines.join("\n");
            }
        }
    }
    return null;
}

function loadContent(content) {
    var Module = module.constructor;
    var m = new Module();
    m._compile(content, "tmp.js");
    return m.exports;
}

function readFile(filePath) {
    return fs.readFileSync(filePath, 'utf8');
}

function loadThings(thingsToLoad, kindOfLoad, funcToCall, fileContent) {
    var content = '';
    for (var i = 0; i < thingsToLoad.length; ++i) {
        var tmp = funcToCall(fileContent, thingsToLoad[i]);
        if (tmp === null) {
            console.error('enable to find ' + kindOfLoad + ' "' + thingsToLoad[i] + '"');
            process.exit(1);
        }
        content += tmp;
        content += 'exports.' + thingsToLoad[i] + ' = ' + thingsToLoad[i] + ';';
    }
    return content;
}

function lookForEntry(entry, data) {
    for (var i = 0; i < data.length; ++i) {
        var allGood = true;
        for (var key in entry) {
            if (!entry.hasOwnProperty(key)) {
                continue;
            }
            var value = data[i][key];
            // To make our life easier, if there is a "parent" type, we add it to the path.
            if (key === 'path' && data[i]['parent'] !== undefined) {
                if (value.length > 0) {
                    value += '::' + data[i]['parent']['name'];
                } else {
                    value = data[i]['parent']['name'];
                }
            }
            if (value !== entry[key]) {
                allGood = false;
                break;
            }
        }
        if (allGood === true) {
            return true;
        }
    }
    return false;
}

function main(argv) {
    if (argv.length !== 3) {
        console.error("Expected toolchain to check as argument (for example 'x86_64-apple-darwin'");
        return 1;
    }
    var toolchain = argv[2];

    var mainJs = readFile("build/" + toolchain + "/doc/main.js");
    var searchIndex = readFile("build/" + toolchain + "/doc/search-index.js").split("\n");
    if (searchIndex[searchIndex.length - 1].length === 0) {
        searchIndex.pop();
    }
    searchIndex.pop();
    searchIndex = loadContent(searchIndex.join("\n") + '\nexports.searchIndex = searchIndex;');
    finalJS = "";

    var arraysToLoad = ["itemTypes"];
    var variablesToLoad = ["MAX_LEV_DISTANCE", "MAX_RESULTS", "TY_PRIMITIVE", "levenshtein_row2"];
    // execQuery first parameter is built in getQuery (which takes in the search input).
    // execQuery last parameter is built in buildIndex.
    // buildIndex requires the hashmap from search-index.
    var functionsToLoad = ["levenshtein", "validateResult", "getQuery", "buildIndex", "execQuery"];

    finalJS += 'window = { "currentCrate": "std" };\n';
    finalJS += loadThings(arraysToLoad, 'array', extractArrayVariable, mainJs);
    finalJS += loadThings(variablesToLoad, 'variable', extractVariable, mainJs);
    finalJS += loadThings(functionsToLoad, 'function', extractFunction, mainJs);

    var loaded = loadContent(finalJS);
    var index = loaded.buildIndex(searchIndex.searchIndex);

    var errors = 0;

    fs.readdirSync(TEST_FOLDER).forEach(function(file) {
        var loadedFile = loadContent(readFile(TEST_FOLDER + file) +
                               'exports.QUERY = QUERY;exports.EXPECTED = EXPECTED;');
        const expected = loadedFile.EXPECTED;
        const query = loadedFile.QUERY;
        var results = loaded.execQuery(loaded.getQuery(query), index);
        process.stdout.write('Checking "' + file + '" ... ');
        var error_text = [];
        for (var key in expected) {
            if (!expected.hasOwnProperty(key)) {
                continue;
            }
            if (!results.hasOwnProperty(key)) {
                error_text.push('==> Unknown key "' + key + '"');
                break;
            }
            var entry = expected[key];
            var found = false;
            for (var i = 0; i < entry.length; ++i) {
                if (lookForEntry(entry[i], results[key]) === true) {
                    found = true;
                } else {
                    error_text.push("==> Result not found in '" + key + "': '" +
                                    JSON.stringify(entry[i]) + "'");
                }
            }
        }
        if (error_text.length !== 0) {
            errors += 1;
            console.error("FAILED");
            console.error(error_text.join("\n"));
        } else {
            console.log("OK");
        }
    });
    return errors;
}

process.exit(main(process.argv));
