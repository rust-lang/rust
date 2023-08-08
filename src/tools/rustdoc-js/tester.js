const fs = require("fs");
const path = require("path");

function loadContent(content) {
    const Module = module.constructor;
    const m = new Module();
    m._compile(content, "tmp.js");
    m.exports.ignore_order = content.indexOf("\n// ignore-order\n") !== -1 ||
        content.startsWith("// ignore-order\n");
    m.exports.exact_check = content.indexOf("\n// exact-check\n") !== -1 ||
        content.startsWith("// exact-check\n");
    m.exports.should_fail = content.indexOf("\n// should-fail\n") !== -1 ||
        content.startsWith("// should-fail\n");
    return m.exports;
}

function readFile(filePath) {
    return fs.readFileSync(filePath, "utf8");
}

function contentToDiffLine(key, value) {
    return `"${key}": "${value}",`;
}

function shouldIgnoreField(fieldName) {
    return fieldName === "query" || fieldName === "correction";
}

// This function is only called when no matching result was found and therefore will only display
// the diff between the two items.
function betterLookingDiff(entry, data) {
    let output = " {\n";
    const spaces = "     ";
    for (const key in entry) {
        if (!Object.prototype.hasOwnProperty.call(entry, key)) {
            continue;
        }
        if (!data || !Object.prototype.hasOwnProperty.call(data, key)) {
            output += "-" + spaces + contentToDiffLine(key, entry[key]) + "\n";
            continue;
        }
        const value = data[key];
        if (value !== entry[key]) {
            output += "-" + spaces + contentToDiffLine(key, entry[key]) + "\n";
            output += "+" + spaces + contentToDiffLine(key, value) + "\n";
        } else {
            output += spaces + contentToDiffLine(key, value) + "\n";
        }
    }
    return output + " }";
}

function lookForEntry(entry, data) {
    return data.findIndex(data_entry => {
        let allGood = true;
        for (const key in entry) {
            if (!Object.prototype.hasOwnProperty.call(entry, key)) {
                continue;
            }
            let value = data_entry[key];
            // To make our life easier, if there is a "parent" type, we add it to the path.
            if (key === "path" && data_entry["parent"] !== undefined) {
                if (value.length > 0) {
                    value += "::" + data_entry["parent"]["name"];
                } else {
                    value = data_entry["parent"]["name"];
                }
            }
            if (value !== entry[key]) {
                allGood = false;
                break;
            }
        }
        return allGood === true;
    });
}

// This function checks if `expected` has all the required fields needed for the checks.
function checkNeededFields(fullPath, expected, error_text, queryName, position) {
    let fieldsToCheck;
    if (fullPath.length === 0) {
        fieldsToCheck = [
            "foundElems",
            "original",
            "returned",
            "userQuery",
            "error",
        ];
    } else if (fullPath.endsWith("elems") || fullPath.endsWith("returned")) {
        fieldsToCheck = [
            "name",
            "fullPath",
            "pathWithoutLast",
            "pathLast",
            "generics",
        ];
    } else if (fullPath.endsWith("generics")) {
        fieldsToCheck = [
            "name",
            "fullPath",
            "pathWithoutLast",
            "pathLast",
            "generics",
        ];
    } else {
        fieldsToCheck = [];
    }
    for (const field of fieldsToCheck) {
        if (!Object.prototype.hasOwnProperty.call(expected, field)) {
            let text = `${queryName}==> Mandatory key \`${field}\` is not present`;
            if (fullPath.length > 0) {
                text += ` in field \`${fullPath}\``;
                if (position !== null) {
                    text += ` (position ${position})`;
                }
            }
            error_text.push(text);
        }
    }
}

function valueCheck(fullPath, expected, result, error_text, queryName) {
    if (Array.isArray(expected)) {
        let i;
        for (i = 0; i < expected.length; ++i) {
            checkNeededFields(fullPath, expected[i], error_text, queryName, i);
            if (i >= result.length) {
                error_text.push(`${queryName}==> EXPECTED has extra value in array from field ` +
                    `\`${fullPath}\` (position ${i}): \`${JSON.stringify(expected[i])}\``);
            } else {
                valueCheck(fullPath + "[" + i + "]", expected[i], result[i], error_text, queryName);
            }
        }
        for (; i < result.length; ++i) {
            error_text.push(`${queryName}==> RESULT has extra value in array from field ` +
                `\`${fullPath}\` (position ${i}): \`${JSON.stringify(result[i])}\` ` +
                "compared to EXPECTED");
        }
    } else if (expected !== null && typeof expected !== "undefined" &&
               expected.constructor == Object) { // eslint-disable-line eqeqeq
        for (const key in expected) {
            if (shouldIgnoreField(key)) {
                continue;
            }
            if (!Object.prototype.hasOwnProperty.call(expected, key)) {
                continue;
            }
            if (!Object.prototype.hasOwnProperty.call(result, key)) {
                error_text.push("==> Unknown key \"" + key + "\"");
                break;
            }
            let result_v = result[key];
            if (result_v !== null && key === "error") {
                result_v.forEach((value, index) => {
                    value = value.split("&nbsp;").join(" ");
                    if (index % 2 === 1) {
                        result_v[index] = "`" + value + "`";
                    } else {
                        result_v[index] = value;
                    }
                });
                result_v = result_v.join("");
            }
            const obj_path = fullPath + (fullPath.length > 0 ? "." : "") + key;
            valueCheck(obj_path, expected[key], result_v, error_text, queryName);
        }
    } else {
        const expectedValue = JSON.stringify(expected);
        const resultValue = JSON.stringify(result);
        if (expectedValue !== resultValue) {
            error_text.push(`${queryName}==> Different values for field \`${fullPath}\`:\n` +
                `EXPECTED: \`${expectedValue}\`\nRESULT:   \`${resultValue}\``);
        }
    }
}

function runParser(query, expected, parseQuery, queryName) {
    const error_text = [];
    checkNeededFields("", expected, error_text, queryName, null);
    if (error_text.length === 0) {
        valueCheck("", expected, parseQuery(query), error_text, queryName);
    }
    return error_text;
}

function runSearch(query, expected, doSearch, loadedFile, queryName) {
    const ignore_order = loadedFile.ignore_order;
    const exact_check = loadedFile.exact_check;

    const results = doSearch(query, loadedFile.FILTER_CRATE);
    const error_text = [];

    for (const key in expected) {
        if (shouldIgnoreField(key)) {
            continue;
        }
        if (!Object.prototype.hasOwnProperty.call(expected, key)) {
            continue;
        }
        if (!Object.prototype.hasOwnProperty.call(results, key)) {
            error_text.push("==> Unknown key \"" + key + "\"");
            break;
        }
        const entry = expected[key];

        if (exact_check && entry.length !== results[key].length) {
            error_text.push(queryName + "==> Expected exactly " + entry.length +
                            " results but found " + results[key].length + " in '" + key + "'");
        }

        let prev_pos = -1;
        entry.forEach((elem, index) => {
            const entry_pos = lookForEntry(elem, results[key]);
            if (entry_pos === -1) {
                error_text.push(queryName + "==> Result not found in '" + key + "': '" +
                                JSON.stringify(elem) + "'");
                // By default, we just compare the two first items.
                let item_to_diff = 0;
                if ((!ignore_order || exact_check) && index < results[key].length) {
                    item_to_diff = index;
                }
                error_text.push("Diff of first error:\n" +
                    betterLookingDiff(elem, results[key][item_to_diff]));
            } else if (exact_check === true && prev_pos + 1 !== entry_pos) {
                error_text.push(queryName + "==> Exact check failed at position " + (prev_pos + 1) +
                                ": expected '" + JSON.stringify(elem) + "' but found '" +
                                JSON.stringify(results[key][index]) + "'");
            } else if (ignore_order === false && entry_pos < prev_pos) {
                error_text.push(queryName + "==> '" + JSON.stringify(elem) + "' was supposed " +
                                "to be before '" + JSON.stringify(results[key][entry_pos]) + "'");
            } else {
                prev_pos = entry_pos;
            }
        });
    }
    return error_text;
}

function runCorrections(query, corrections, getCorrections, loadedFile) {
    const qc = getCorrections(query, loadedFile.FILTER_CRATE);
    const error_text = [];

    if (corrections === null) {
        if (qc !== null) {
            error_text.push(`==> expected = null, found = ${qc}`);
        }
        return error_text;
    }

    if (qc !== corrections.toLowerCase()) {
        error_text.push(`==> expected = ${corrections}, found = ${qc}`);
    }

    return error_text;
}

function checkResult(error_text, loadedFile, displaySuccess) {
    if (error_text.length === 0 && loadedFile.should_fail === true) {
        console.log("FAILED");
        console.log("==> Test was supposed to fail but all items were found...");
    } else if (error_text.length !== 0 && loadedFile.should_fail === false) {
        console.log("FAILED");
        console.log(error_text.join("\n"));
    } else {
        if (displaySuccess) {
            console.log("OK");
        }
        return 0;
    }
    return 1;
}

function runCheckInner(callback, loadedFile, entry, getCorrections, extra) {
    if (typeof entry.query !== "string") {
        console.log("FAILED");
        console.log("==> Missing `query` field");
        return false;
    }
    let error_text = callback(entry.query, entry, extra ? "[ query `" + entry.query + "`]" : "");
    if (checkResult(error_text, loadedFile, false) !== 0) {
        return false;
    }
    if (entry.correction !== undefined) {
        error_text = runCorrections(entry.query, entry.correction, getCorrections, loadedFile);
        if (checkResult(error_text, loadedFile, false) !== 0) {
            return false;
        }
    }
    return true;
}

function runCheck(loadedFile, key, getCorrections, callback) {
    const expected = loadedFile[key];

    if (Array.isArray(expected)) {
        for (const entry of expected) {
            if (!runCheckInner(callback, loadedFile, entry, getCorrections, true)) {
                return 1;
            }
        }
    } else if (!runCheckInner(callback, loadedFile, expected, getCorrections, false)) {
        return 1;
    }
    console.log("OK");
    return 0;
}

function hasCheck(content, checkName) {
    return content.startsWith(`const ${checkName}`) || content.includes(`\nconst ${checkName}`);
}

function runChecks(testFile, doSearch, parseQuery, getCorrections) {
    let checkExpected = false;
    let checkParsed = false;
    let testFileContent = readFile(testFile);

    if (testFileContent.indexOf("FILTER_CRATE") !== -1) {
        testFileContent += "exports.FILTER_CRATE = FILTER_CRATE;";
    } else {
        testFileContent += "exports.FILTER_CRATE = null;";
    }

    if (hasCheck(testFileContent, "EXPECTED")) {
        testFileContent += "exports.EXPECTED = EXPECTED;";
        checkExpected = true;
    }
    if (hasCheck(testFileContent, "PARSED")) {
        testFileContent += "exports.PARSED = PARSED;";
        checkParsed = true;
    }
    if (!checkParsed && !checkExpected) {
        console.log("FAILED");
        console.log("==> At least `PARSED` or `EXPECTED` is needed!");
        return 1;
    }

    const loadedFile = loadContent(testFileContent);
    let res = 0;

    if (checkExpected) {
        res += runCheck(loadedFile, "EXPECTED", getCorrections, (query, expected, text) => {
            return runSearch(query, expected, doSearch, loadedFile, text);
        });
    }
    if (checkParsed) {
        res += runCheck(loadedFile, "PARSED", getCorrections, (query, expected, text) => {
            return runParser(query, expected, parseQuery, text);
        });
    }
    return res;
}

/**
 * Load searchNNN.js and search-indexNNN.js.
 *
 * @param {string} doc_folder      - Path to a folder generated by running rustdoc
 * @param {string} resource_suffix - Version number between filename and .js, e.g. "1.59.0"
 * @returns {Object}               - Object containing keys: `doSearch`, which runs a search
 *   with the loaded index and returns a table of results; `parseQuery`, which is the
 *   `parseQuery` function exported from the search module; and `getCorrections`, which runs
 *   a search but returns type name corrections instead of results.
 */
function loadSearchJS(doc_folder, resource_suffix) {
    const searchIndexJs = path.join(doc_folder, "search-index" + resource_suffix + ".js");
    const searchIndex = require(searchIndexJs);

    const staticFiles = path.join(doc_folder, "static.files");
    const searchJs = fs.readdirSync(staticFiles).find(f => f.match(/search.*\.js$/));
    const searchModule = require(path.join(staticFiles, searchJs));
    const searchWords = searchModule.initSearch(searchIndex.searchIndex);

    return {
        doSearch: function(queryStr, filterCrate, currentCrate) {
            return searchModule.execQuery(searchModule.parseQuery(queryStr), searchWords,
                filterCrate, currentCrate);
        },
        getCorrections: function(queryStr, filterCrate, currentCrate) {
            const parsedQuery = searchModule.parseQuery(queryStr);
            searchModule.execQuery(parsedQuery, searchWords, filterCrate, currentCrate);
            return parsedQuery.correction;
        },
        parseQuery: searchModule.parseQuery,
    };
}

function showHelp() {
    console.log("rustdoc-js options:");
    console.log("  --doc-folder [PATH]        : location of the generated doc folder");
    console.log("  --help                     : show this message then quit");
    console.log("  --crate-name [STRING]      : crate name to be used");
    console.log("  --test-file [PATHs]        : location of the JS test files (can be called " +
                "multiple times)");
    console.log("  --test-folder [PATH]       : location of the JS tests folder");
    console.log("  --resource-suffix [STRING] : suffix to refer to the correct files");
}

function parseOptions(args) {
    const opts = {
        "crate_name": "",
        "resource_suffix": "",
        "doc_folder": "",
        "test_folder": "",
        "test_file": [],
    };
    const correspondences = {
        "--resource-suffix": "resource_suffix",
        "--doc-folder": "doc_folder",
        "--test-folder": "test_folder",
        "--test-file": "test_file",
        "--crate-name": "crate_name",
    };

    for (let i = 0; i < args.length; ++i) {
        const arg = args[i];
        if (Object.prototype.hasOwnProperty.call(correspondences, arg)) {
            i += 1;
            if (i >= args.length) {
                console.log("Missing argument after `" + arg + "` option.");
                return null;
            }
            const arg_value = args[i];
            if (arg !== "--test-file") {
                opts[correspondences[arg]] = arg_value;
            } else {
                opts[correspondences[arg]].push(arg_value);
            }
        } else if (arg === "--help") {
            showHelp();
            process.exit(0);
        } else {
            console.log("Unknown option `" + arg + "`.");
            console.log("Use `--help` to see the list of options");
            return null;
        }
    }
    if (opts["doc_folder"].length < 1) {
        console.log("Missing `--doc-folder` option.");
    } else if (opts["crate_name"].length < 1) {
        console.log("Missing `--crate-name` option.");
    } else if (opts["test_folder"].length < 1 && opts["test_file"].length < 1) {
        console.log("At least one of `--test-folder` or `--test-file` option is required.");
    } else {
        return opts;
    }
    return null;
}

function main(argv) {
    const opts = parseOptions(argv.slice(2));
    if (opts === null) {
        return 1;
    }

    const parseAndSearch = loadSearchJS(
        opts["doc_folder"],
        opts["resource_suffix"]
    );
    let errors = 0;

    const doSearch = function(queryStr, filterCrate) {
        return parseAndSearch.doSearch(queryStr, filterCrate, opts["crate_name"]);
    };
    const getCorrections = function(queryStr, filterCrate) {
        return parseAndSearch.getCorrections(queryStr, filterCrate, opts["crate_name"]);
    };

    if (opts["test_file"].length !== 0) {
        opts["test_file"].forEach(file => {
            process.stdout.write(`Testing ${file} ... `);
            errors += runChecks(file, doSearch, parseAndSearch.parseQuery, getCorrections);
        });
    } else if (opts["test_folder"].length !== 0) {
        fs.readdirSync(opts["test_folder"]).forEach(file => {
            if (!file.endsWith(".js")) {
                return;
            }
            process.stdout.write(`Testing ${file} ... `);
            errors += runChecks(path.join(opts["test_folder"], file), doSearch,
                    parseAndSearch.parseQuery, getCorrections);
        });
    }
    return errors > 0 ? 1 : 0;
}

process.exit(main(process.argv));
