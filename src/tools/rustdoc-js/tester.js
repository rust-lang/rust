/* global globalThis */

const fs = require("fs");
const path = require("path");
const { isGeneratorObject } = require("util/types");

function arrayToCode(array) {
    return array.map((value, index) => {
        value = value.split("&nbsp;").join(" ");
        return (index % 2 === 1) ? ("`" + value + "`") : value;
    }).join("");
}

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
    if (typeof value === "object" && !Array.isArray(value) && value !== null) {
        const out = Object.entries(value)
            .filter(([subKey, _]) => ["path", "name"].includes(subKey))
            .map(([subKey, subValue]) => `"${subKey}": ${JSON.stringify(subValue)}`)
            .join(", ");
        return `"${key}": ${out},`;
    }
    return `"${key}": ${JSON.stringify(value)},`;
}

function shouldIgnoreField(fieldName) {
    return fieldName === "query" || fieldName === "correction" ||
        fieldName === "proposeCorrectionFrom" ||
        fieldName === "proposeCorrectionTo";
}

function valueMapper(key, testOutput) {
    let value = testOutput[key];
    // To make our life easier, if there is a "parent" type, we add it to the path.
    if (key === "path") {
        if (testOutput["parent"]) {
            if (value.length > 0) {
                value += "::" + testOutput["parent"]["name"];
            } else {
                value = testOutput["parent"]["name"];
            }
        }
    }
    return value;
}

// This function is only called when no matching result was found and therefore will only display
// the diff between the two items.
function betterLookingDiff(expected, testOutput) {
    let output = " {\n";
    const spaces = "    ";
    for (const key in expected) {
        if (!Object.prototype.hasOwnProperty.call(expected, key)) {
            continue;
        }
        if (!testOutput || !Object.prototype.hasOwnProperty.call(testOutput, key)) {
            output += "-" + spaces + contentToDiffLine(key, expected[key]) + "\n";
            continue;
        }
        const value = valueMapper(key, testOutput);
        if (value !== expected[key]) {
            output += "-" + spaces + contentToDiffLine(key, expected[key]) + "\n";
            output += "+" + spaces + contentToDiffLine(key, value) + "\n";
        } else {
            output += spaces + " " + contentToDiffLine(key, value) + "\n";
        }
    }
    return output + " }";
}

function lookForEntry(expected, testOutput) {
    return testOutput.findIndex(testOutputEntry => {
        let allGood = true;
        for (const key in expected) {
            if (!Object.prototype.hasOwnProperty.call(expected, key)) {
                continue;
            }
            const value = valueMapper(key, testOutputEntry);
            if (value !== expected[key]) {
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
    if (Array.isArray(expected) && result instanceof Map) {
        const expected_set = new Set();
        for (const [key, expected_value] of expected) {
            expected_set.add(key);
            checkNeededFields(fullPath, expected_value, error_text, queryName, key);
            if (result.has(key)) {
                valueCheck(
                    fullPath + "[" + key + "]",
                    expected_value,
                    result.get(key),
                    error_text,
                    queryName,
                );
            } else {
                error_text.push(`${queryName}==> EXPECTED has extra key in map from field ` +
                    `\`${fullPath}\` (key ${key}): \`${JSON.stringify(expected_value)}\``);
            }
        }
        for (const [key, result_value] of result.entries()) {
            if (!expected_set.has(key)) {
                error_text.push(`${queryName}==> EXPECTED missing key in map from field ` +
                    `\`${fullPath}\` (key ${key}): \`${JSON.stringify(result_value)}\``);
            }
        }
    } else if (Array.isArray(expected)) {
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
                if (!result_v.forEach) {
                    throw result_v;
                }
                result_v = arrayToCode(result_v);
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

async function runSearch(query, expected, doSearch, loadedFile, queryName) {
    const ignore_order = loadedFile.ignore_order;
    const exact_check = loadedFile.exact_check;

    const { resultsTable } = await doSearch(query, loadedFile.FILTER_CRATE);
    const error_text = [];

    for (const key in expected) {
        if (shouldIgnoreField(key)) {
            continue;
        }
        if (!Object.prototype.hasOwnProperty.call(expected, key)) {
            continue;
        }
        if (!Object.prototype.hasOwnProperty.call(resultsTable, key)) {
            error_text.push("==> Unknown key \"" + key + "\"");
            break;
        }
        const entry = expected[key];

        if (exact_check && entry.length !== resultsTable[key].length) {
            error_text.push(queryName + "==> Expected exactly " + entry.length +
                            " results but found " + resultsTable[key].length + " in '" + key + "'");
        }

        let prev_pos = -1;
        for (const [index, elem] of entry.entries()) {
            const entry_pos = lookForEntry(elem, resultsTable[key]);
            if (entry_pos === -1) {
                error_text.push(queryName + "==> Result not found in '" + key + "': '" +
                                JSON.stringify(elem) + "'");
                // By default, we just compare the two first items.
                let item_to_diff = 0;
                if ((!ignore_order || exact_check) && index < resultsTable[key].length) {
                    item_to_diff = index;
                }
                error_text.push("Diff of first error:\n" +
                    betterLookingDiff(elem, resultsTable[key][item_to_diff]));
            } else if (exact_check === true && prev_pos + 1 !== entry_pos) {
                error_text.push(queryName + "==> Exact check failed at position " + (prev_pos + 1) +
                                ": expected '" + JSON.stringify(elem) + "' but found '" +
                                JSON.stringify(resultsTable[key][index]) + "'");
            } else if (ignore_order === false && entry_pos < prev_pos) {
                error_text.push(queryName + "==> '" +
                                JSON.stringify(elem) + "' was supposed to be before '" +
                                JSON.stringify(resultsTable[key][prev_pos]) + "'");
            } else {
                prev_pos = entry_pos;
            }
        }
    }
    return error_text;
}

async function runCorrections(query, corrections, doSearch, loadedFile) {
    const { parsedQuery } = await doSearch(query, loadedFile.FILTER_CRATE);
    const qc = parsedQuery.correction;
    const error_text = [];

    if (corrections === null) {
        if (qc !== null) {
            error_text.push(`==> [correction] expected = null, found = ${qc}`);
        }
        return error_text;
    }

    if (qc.toLowerCase() !== corrections.toLowerCase()) {
        error_text.push(`==> [correction] expected = ${corrections}, found = ${qc}`);
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

async function runCheckInner(callback, loadedFile, entry, extra, doSearch) {
    if (typeof entry.query !== "string") {
        console.log("FAILED");
        console.log("==> Missing `query` field");
        return false;
    }
    let error_text = await callback(
        entry.query,
        entry,
        extra ? "[ query `" + entry.query + "`]" : "",
    );
    if (checkResult(error_text, loadedFile, false) !== 0) {
        return false;
    }
    if (entry.correction !== undefined) {
        error_text = await runCorrections(
            entry.query,
            entry.correction,
            doSearch,
            loadedFile,
        );
        if (checkResult(error_text, loadedFile, false) !== 0) {
            return false;
        }
    }
    return true;
}

async function runCheck(loadedFile, key, doSearch, callback) {
    const expected = loadedFile[key];

    if (Array.isArray(expected)) {
        for (const entry of expected) {
            if (!await runCheckInner(callback, loadedFile, entry, true, doSearch)) {
                return 1;
            }
        }
    } else if (!await runCheckInner(callback, loadedFile, expected, false, doSearch)) {
        return 1;
    }
    console.log("OK");
    return 0;
}

function hasCheck(content, checkName) {
    return content.startsWith(`const ${checkName}`) || content.includes(`\nconst ${checkName}`);
}

async function runChecks(testFile, doSearch, parseQuery) {
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
        res += await runCheck(loadedFile, "EXPECTED", doSearch, (query, expected, text) => {
            return runSearch(query, expected, doSearch, loadedFile, text);
        });
    }
    if (checkParsed) {
        res += await runCheck(loadedFile, "PARSED", doSearch, (query, expected, text) => {
            return runParser(query, expected, parseQuery, text);
        });
    }
    return res;
}

function mostRecentMatch(staticFiles, regex) {
    const matchingEntries = fs.readdirSync(staticFiles)
        .filter(f => f.match(regex))
        .map(f => {
            const stats = fs.statSync(path.join(staticFiles, f));
            return {
                path: f,
                time: stats.mtimeMs,
            };
        });
    if (matchingEntries.length === 0) {
        throw "No static file matching regex";
    }
    // We sort entries in descending order.
    matchingEntries.sort((a, b) => b.time - a.time);
    return matchingEntries[0].path;
}

/**
 * Load searchNNN.js and search-indexNNN.js.
 *
 * @param {string} doc_folder      - Path to a folder generated by running rustdoc
 * @param {string} resource_suffix - Version number between filename and .js, e.g. "1.59.0"
 * @returns {Object}               - Object containing keys: `doSearch`, which runs a search
 *   with the loaded index and returns a table of results; `parseQuery`, which is the
 *   `parseQuery` function exported from the search module, which runs
 *   a search but returns type name corrections instead of results.
 */
async function loadSearchJS(doc_folder, resource_suffix) {
    const staticFiles = path.join(doc_folder, "static.files");
    const stringdexJs = mostRecentMatch(staticFiles, /stringdex.*\.js$/);
    const stringdexModule = require(path.join(staticFiles, stringdexJs));
    const searchJs = mostRecentMatch(staticFiles, /search-[0-9a-f]{8}.*\.js$/);
    const searchModule = require(path.join(staticFiles, searchJs));
    globalThis.nonnull = (x, msg) => {
        if (x === null) {
            throw (msg || "unexpected null value!");
        } else {
            return x;
        }
    };
    const { docSearch, DocSearch } = await searchModule.initSearch(
        stringdexModule.Stringdex,
        stringdexModule.RoaringBitmap,
        {
            loadRoot: callbacks => {
                for (const key in callbacks) {
                    if (Object.hasOwn(callbacks, key)) {
                        globalThis[key] = callbacks[key];
                    }
                }
                const rootJs = readFile(path.join(doc_folder, "search.index/root" +
                    resource_suffix + ".js"));
                eval(rootJs);
            },
            loadTreeByHash: hashHex => {
                const shardJs = readFile(path.join(doc_folder, "search.index/" + hashHex + ".js"));
                eval(shardJs);
            },
            loadDataByNameAndHash: (name, hashHex) => {
                const shardJs = readFile(path.join(doc_folder, "search.index/" + name + "/" +
                    hashHex + ".js"));
                eval(shardJs);
            },
        },
    );
    return {
        doSearch: async function(queryStr, filterCrate, currentCrate) {
            const parsedQuery = DocSearch.parseQuery(queryStr);
            const result = await docSearch.execQuery(parsedQuery, filterCrate, currentCrate);
            const resultsTable = {};
            for (const tab in result) {
                if (!Object.prototype.hasOwnProperty.call(result, tab)) {
                    continue;
                }
                if (!isGeneratorObject(result[tab])) {
                    continue;
                }
                resultsTable[tab] = [];
                for await (const entry of result[tab]) {
                    const flatEntry = Object.assign({
                        crate: entry.item.crate,
                        name: entry.item.name,
                        path: entry.item.modulePath,
                        exactPath: entry.item.exactModulePath,
                        ty: entry.item.ty,
                    }, entry);
                    for (const key in entry) {
                        if (!Object.prototype.hasOwnProperty.call(entry, key)) {
                            continue;
                        }
                        if (key === "desc" && entry.desc !== null) {
                            flatEntry.desc = await entry.desc;
                        } else if (key === "displayTypeSignature" &&
                            entry.displayTypeSignature !== null
                        ) {
                            flatEntry.displayTypeSignature = await entry.displayTypeSignature;
                            const {
                                type,
                                mappedNames,
                                whereClause,
                            } = flatEntry.displayTypeSignature;
                            flatEntry.displayType = arrayToCode(type);
                            flatEntry.displayMappedNames = [...mappedNames.entries()]
                                .map(([name, qname]) => {
                                    return `${name} = ${qname}`;
                                }).join(", ");
                            flatEntry.displayWhereClause = [...whereClause.entries()]
                                .flatMap(([name, value]) => {
                                    if (value.length === 0) {
                                        return [];
                                    }
                                    return [`${name}: ${arrayToCode(value)}`];
                                }).join(", ");
                        }
                    }
                    resultsTable[tab].push(flatEntry);
                }
            }
            return { resultsTable, parsedQuery };
        },
        parseQuery: DocSearch.parseQuery,
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

async function main(argv) {
    const opts = parseOptions(argv.slice(2));
    if (opts === null) {
        return 1;
    }

    const parseAndSearch = await loadSearchJS(
        opts["doc_folder"],
        opts["resource_suffix"],
    );
    let errors = 0;

    const doSearch = function(queryStr, filterCrate) {
        return parseAndSearch.doSearch(queryStr, filterCrate, opts["crate_name"]);
    };

    if (opts["test_file"].length !== 0) {
        for (const file of opts["test_file"]) {
            process.stdout.write(`Testing ${file} ... `);
            errors += await runChecks(file, doSearch, parseAndSearch.parseQuery);
        }
    } else if (opts["test_folder"].length !== 0) {
        for (const file of fs.readdirSync(opts["test_folder"])) {
            if (!file.endsWith(".js")) {
                continue;
            }
            process.stdout.write(`Testing ${file} ... `);
            errors += await runChecks(path.join(opts["test_folder"], file), doSearch,
                    parseAndSearch.parseQuery);
        }
    }
    return errors > 0 ? 1 : 0;
}

main(process.argv).catch(e => {
    console.log(e);
    process.exit(1);
}).then(x => process.exit(x));

process.on("beforeExit", () => {
    console.log("process did not complete");
    process.exit(1);
});
