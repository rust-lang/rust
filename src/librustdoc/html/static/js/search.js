/* global addClass, getNakedUrl, getSettingValue, hasOwnPropertyRustdoc, initSearch, onEach */
/* global onEachLazy, removeClass, searchState, updateLocalStorage */

(function() {
// This mapping table should match the discriminants of
// `rustdoc::html::item_type::ItemType` type in Rust.
var itemTypes = ["mod",
                    "externcrate",
                    "import",
                    "struct",
                    "enum",
                    "fn",
                    "type",
                    "static",
                    "trait",
                    "impl",
                    "tymethod",
                    "method",
                    "structfield",
                    "variant",
                    "macro",
                    "primitive",
                    "associatedtype",
                    "constant",
                    "associatedconstant",
                    "union",
                    "foreigntype",
                    "keyword",
                    "existential",
                    "attr",
                    "derive",
                    "traitalias"];

// used for special search precedence
var TY_PRIMITIVE = itemTypes.indexOf("primitive");
var TY_KEYWORD = itemTypes.indexOf("keyword");

// In the search display, allows to switch between tabs.
function printTab(nb) {
    if (nb === 0 || nb === 1 || nb === 2) {
        searchState.currentTab = nb;
    }
    var nb_copy = nb;
    onEachLazy(document.getElementById("titles").childNodes, function(elem) {
        if (nb_copy === 0) {
            addClass(elem, "selected");
        } else {
            removeClass(elem, "selected");
        }
        nb_copy -= 1;
    });
    onEachLazy(document.getElementById("results").childNodes, function(elem) {
        if (nb === 0) {
            addClass(elem, "active");
        } else {
            removeClass(elem, "active");
        }
        nb -= 1;
    });
}

function removeEmptyStringsFromArray(x) {
    for (var i = 0, len = x.length; i < len; ++i) {
        if (x[i] === "") {
            x.splice(i, 1);
            i -= 1;
        }
    }
}

/**
 * A function to compute the Levenshtein distance between two strings
 * Licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported
 * Full License can be found at http://creativecommons.org/licenses/by-sa/3.0/legalcode
 * This code is an unmodified version of the code written by Marco de Wit
 * and was found at https://stackoverflow.com/a/18514751/745719
 */
var levenshtein_row2 = [];
function levenshtein(s1, s2) {
    if (s1 === s2) {
        return 0;
    }
    var s1_len = s1.length, s2_len = s2.length;
    if (s1_len && s2_len) {
        var i1 = 0, i2 = 0, a, b, c, c2, row = levenshtein_row2;
        while (i1 < s1_len) {
            row[i1] = ++i1;
        }
        while (i2 < s2_len) {
            c2 = s2.charCodeAt(i2);
            a = i2;
            ++i2;
            b = i2;
            for (i1 = 0; i1 < s1_len; ++i1) {
                c = a + (s1.charCodeAt(i1) !== c2 ? 1 : 0);
                a = row[i1];
                b = b < a ? (b < c ? b + 1 : c) : (a < c ? a + 1 : c);
                row[i1] = b;
            }
        }
        return b;
    }
    return s1_len + s2_len;
}

window.initSearch = function(rawSearchIndex) {
    var MAX_LEV_DISTANCE = 3;
    var MAX_RESULTS = 200;
    var GENERICS_DATA = 2;
    var NAME = 0;
    var INPUTS_DATA = 0;
    var OUTPUT_DATA = 1;
    var NO_TYPE_FILTER = -1;
    var currentResults, index, searchIndex;
    var ALIASES = {};
    var params = searchState.getQueryStringParams();

    // Populate search bar with query string search term when provided,
    // but only if the input bar is empty. This avoid the obnoxious issue
    // where you start trying to do a search, and the index loads, and
    // suddenly your search is gone!
    if (searchState.input.value === "") {
        searchState.input.value = params.search || "";
    }

    /**
     * Executes the query and builds an index of results
     * @param  {[Object]} query      [The user query]
     * @param  {[type]} searchWords  [The list of search words to query
     *                                against]
     * @param  {[type]} filterCrates [Crate to search in if defined]
     * @return {[type]}              [A search index of results]
     */
    function execQuery(query, searchWords, filterCrates) {
        function itemTypeFromName(typename) {
            for (var i = 0, len = itemTypes.length; i < len; ++i) {
                if (itemTypes[i] === typename) {
                    return i;
                }
            }
            return NO_TYPE_FILTER;
        }

        var valLower = query.query.toLowerCase(),
            val = valLower,
            typeFilter = itemTypeFromName(query.type),
            results = {}, results_in_args = {}, results_returned = {},
            split = valLower.split("::");

        removeEmptyStringsFromArray(split);

        function transformResults(results) {
            var out = [];
            for (var i = 0, len = results.length; i < len; ++i) {
                if (results[i].id > -1) {
                    var obj = searchIndex[results[i].id];
                    obj.lev = results[i].lev;
                    var res = buildHrefAndPath(obj);
                    obj.displayPath = pathSplitter(res[0]);
                    obj.fullPath = obj.displayPath + obj.name;
                    // To be sure than it some items aren't considered as duplicate.
                    obj.fullPath += "|" + obj.ty;
                    obj.href = res[1];
                    out.push(obj);
                    if (out.length >= MAX_RESULTS) {
                        break;
                    }
                }
            }
            return out;
        }

        function sortResults(results, isType) {
            var ar = [];
            for (var entry in results) {
                if (hasOwnPropertyRustdoc(results, entry)) {
                    ar.push(results[entry]);
                }
            }
            results = ar;
            var i, len, result;
            for (i = 0, len = results.length; i < len; ++i) {
                result = results[i];
                result.word = searchWords[result.id];
                result.item = searchIndex[result.id] || {};
            }
            // if there are no results then return to default and fail
            if (results.length === 0) {
                return [];
            }

            results.sort(function(aaa, bbb) {
                var a, b;

                // sort by exact match with regard to the last word (mismatch goes later)
                a = (aaa.word !== val);
                b = (bbb.word !== val);
                if (a !== b) { return a - b; }

                // Sort by non levenshtein results and then levenshtein results by the distance
                // (less changes required to match means higher rankings)
                a = (aaa.lev);
                b = (bbb.lev);
                if (a !== b) { return a - b; }

                // sort by crate (non-current crate goes later)
                a = (aaa.item.crate !== window.currentCrate);
                b = (bbb.item.crate !== window.currentCrate);
                if (a !== b) { return a - b; }

                // sort by item name length (longer goes later)
                a = aaa.word.length;
                b = bbb.word.length;
                if (a !== b) { return a - b; }

                // sort by item name (lexicographically larger goes later)
                a = aaa.word;
                b = bbb.word;
                if (a !== b) { return (a > b ? +1 : -1); }

                // sort by index of keyword in item name (no literal occurrence goes later)
                a = (aaa.index < 0);
                b = (bbb.index < 0);
                if (a !== b) { return a - b; }
                // (later literal occurrence, if any, goes later)
                a = aaa.index;
                b = bbb.index;
                if (a !== b) { return a - b; }

                // special precedence for primitive and keyword pages
                if ((aaa.item.ty === TY_PRIMITIVE && bbb.item.ty !== TY_KEYWORD) ||
                    (aaa.item.ty === TY_KEYWORD && bbb.item.ty !== TY_PRIMITIVE)) {
                    return -1;
                }
                if ((bbb.item.ty === TY_PRIMITIVE && aaa.item.ty !== TY_PRIMITIVE) ||
                    (bbb.item.ty === TY_KEYWORD && aaa.item.ty !== TY_KEYWORD)) {
                    return 1;
                }

                // sort by description (no description goes later)
                a = (aaa.item.desc === "");
                b = (bbb.item.desc === "");
                if (a !== b) { return a - b; }

                // sort by type (later occurrence in `itemTypes` goes later)
                a = aaa.item.ty;
                b = bbb.item.ty;
                if (a !== b) { return a - b; }

                // sort by path (lexicographically larger goes later)
                a = aaa.item.path;
                b = bbb.item.path;
                if (a !== b) { return (a > b ? +1 : -1); }

                // que sera, sera
                return 0;
            });

            for (i = 0, len = results.length; i < len; ++i) {
                result = results[i];

                // this validation does not make sense when searching by types
                if (result.dontValidate) {
                    continue;
                }
                var name = result.item.name.toLowerCase(),
                    path = result.item.path.toLowerCase(),
                    parent = result.item.parent;

                if (!isType && !validateResult(name, path, split, parent)) {
                    result.id = -1;
                }
            }
            return transformResults(results);
        }

        function extractGenerics(val) {
            val = val.toLowerCase();
            if (val.indexOf("<") !== -1) {
                var values = val.substring(val.indexOf("<") + 1, val.lastIndexOf(">"));
                return {
                    name: val.substring(0, val.indexOf("<")),
                    generics: values.split(/\s*,\s*/),
                };
            }
            return {
                name: val,
                generics: [],
            };
        }

        function checkGenerics(obj, val) {
            // The names match, but we need to be sure that all generics kinda
            // match as well.
            var tmp_lev, elem_name;
            if (val.generics.length > 0) {
                if (obj.length > GENERICS_DATA &&
                      obj[GENERICS_DATA].length >= val.generics.length) {
                    var elems = Object.create(null);
                    var elength = obj[GENERICS_DATA].length;
                    for (var x = 0; x < elength; ++x) {
                        if (!elems[obj[GENERICS_DATA][x][NAME]]) {
                            elems[obj[GENERICS_DATA][x][NAME]] = 0;
                        }
                        elems[obj[GENERICS_DATA][x][NAME]] += 1;
                    }
                    var total = 0;
                    var done = 0;
                    // We need to find the type that matches the most to remove it in order
                    // to move forward.
                    var vlength = val.generics.length;
                    for (x = 0; x < vlength; ++x) {
                        var lev = MAX_LEV_DISTANCE + 1;
                        var firstGeneric = val.generics[x];
                        var match = null;
                        if (elems[firstGeneric]) {
                            match = firstGeneric;
                            lev = 0;
                        } else {
                            for (elem_name in elems) {
                                tmp_lev = levenshtein(elem_name, firstGeneric);
                                if (tmp_lev < lev) {
                                    lev = tmp_lev;
                                    match = elem_name;
                                }
                            }
                        }
                        if (match !== null) {
                            elems[match] -= 1;
                            if (elems[match] == 0) {
                                delete elems[match];
                            }
                            total += lev;
                            done += 1;
                        } else {
                            return MAX_LEV_DISTANCE + 1;
                        }
                    }
                    return Math.ceil(total / done);
                }
            }
            return MAX_LEV_DISTANCE + 1;
        }

        // Check for type name and type generics (if any).
        function checkType(obj, val, literalSearch) {
            var lev_distance = MAX_LEV_DISTANCE + 1;
            var tmp_lev = MAX_LEV_DISTANCE + 1;
            var len, x, firstGeneric;
            if (obj[NAME] === val.name) {
                if (literalSearch) {
                    if (val.generics && val.generics.length !== 0) {
                        if (obj.length > GENERICS_DATA &&
                             obj[GENERICS_DATA].length > 0) {
                            var elems = Object.create(null);
                            len = obj[GENERICS_DATA].length;
                            for (x = 0; x < len; ++x) {
                                if (!elems[obj[GENERICS_DATA][x][NAME]]) {
                                    elems[obj[GENERICS_DATA][x][NAME]] = 0;
                                }
                                elems[obj[GENERICS_DATA][x][NAME]] += 1;
                            }

                            var allFound = true;
                            len = val.generics.length;
                            for (x = 0; x < len; ++x) {
                                firstGeneric = val.generics[x];
                                if (elems[firstGeneric]) {
                                    elems[firstGeneric] -= 1;
                                } else {
                                    allFound = false;
                                    break;
                                }
                            }
                            if (allFound) {
                                return true;
                            }
                        }
                        return false;
                    }
                    return true;
                } else {
                    // If the type has generics but don't match, then it won't return at this point.
                    // Otherwise, `checkGenerics` will return 0 and it'll return.
                    if (obj.length > GENERICS_DATA && obj[GENERICS_DATA].length !== 0) {
                        tmp_lev = checkGenerics(obj, val);
                        if (tmp_lev <= MAX_LEV_DISTANCE) {
                            return tmp_lev;
                        }
                    }
                }
            } else if (literalSearch) {
                if ((!val.generics || val.generics.length === 0) &&
                      obj.length > GENERICS_DATA && obj[GENERICS_DATA].length > 0) {
                    return obj[GENERICS_DATA].some(
                        function(gen) {
                            return gen[NAME] === val.name;
                        });
                }
                return false;
            }
            lev_distance = Math.min(levenshtein(obj[NAME], val.name), lev_distance);
            if (lev_distance <= MAX_LEV_DISTANCE) {
                // The generics didn't match but the name kinda did so we give it
                // a levenshtein distance value that isn't *this* good so it goes
                // into the search results but not too high.
                lev_distance = Math.ceil((checkGenerics(obj, val) + lev_distance) / 2);
            }
            if (obj.length > GENERICS_DATA && obj[GENERICS_DATA].length > 0) {
                // We can check if the type we're looking for is inside the generics!
                var olength = obj[GENERICS_DATA].length;
                for (x = 0; x < olength; ++x) {
                    tmp_lev = Math.min(levenshtein(obj[GENERICS_DATA][x][NAME], val.name), tmp_lev);
                }
                if (tmp_lev !== 0) {
                    // If we didn't find a good enough result, we go check inside the generics of
                    // the generics.
                    for (x = 0; x < olength && tmp_lev !== 0; ++x) {
                        tmp_lev = Math.min(
                            checkType(obj[GENERICS_DATA][x], val, literalSearch),
                            tmp_lev
                        );
                    }
                }
            }
            // Now whatever happens, the returned distance is "less good" so we should mark it
            // as such, and so we add 1 to the distance to make it "less good".
            return Math.min(lev_distance, tmp_lev) + 1;
        }

        function findArg(obj, val, literalSearch, typeFilter) {
            var lev_distance = MAX_LEV_DISTANCE + 1;

            if (obj && obj.type && obj.type[INPUTS_DATA] && obj.type[INPUTS_DATA].length > 0) {
                var length = obj.type[INPUTS_DATA].length;
                for (var i = 0; i < length; i++) {
                    var tmp = obj.type[INPUTS_DATA][i];
                    if (!typePassesFilter(typeFilter, tmp[1])) {
                        continue;
                    }
                    tmp = checkType(tmp, val, literalSearch);
                    if (literalSearch) {
                        if (tmp) {
                            return true;
                        }
                        continue;
                    }
                    lev_distance = Math.min(tmp, lev_distance);
                    if (lev_distance === 0) {
                        return 0;
                    }
                }
            }
            return literalSearch ? false : lev_distance;
        }

        function checkReturned(obj, val, literalSearch, typeFilter) {
            var lev_distance = MAX_LEV_DISTANCE + 1;

            if (obj && obj.type && obj.type.length > OUTPUT_DATA) {
                var ret = obj.type[OUTPUT_DATA];
                if (typeof ret[0] === "string") {
                    ret = [ret];
                }
                for (var x = 0, len = ret.length; x < len; ++x) {
                    var tmp = ret[x];
                    if (!typePassesFilter(typeFilter, tmp[1])) {
                        continue;
                    }
                    tmp = checkType(tmp, val, literalSearch);
                    if (literalSearch) {
                        if (tmp) {
                            return true;
                        }
                        continue;
                    }
                    lev_distance = Math.min(tmp, lev_distance);
                    if (lev_distance === 0) {
                        return 0;
                    }
                }
            }
            return literalSearch ? false : lev_distance;
        }

        function checkPath(contains, lastElem, ty) {
            if (contains.length === 0) {
                return 0;
            }
            var ret_lev = MAX_LEV_DISTANCE + 1;
            var path = ty.path.split("::");

            if (ty.parent && ty.parent.name) {
                path.push(ty.parent.name.toLowerCase());
            }

            var length = path.length;
            var clength = contains.length;
            if (clength > length) {
                return MAX_LEV_DISTANCE + 1;
            }
            for (var i = 0; i < length; ++i) {
                if (i + clength > length) {
                    break;
                }
                var lev_total = 0;
                var aborted = false;
                for (var x = 0; x < clength; ++x) {
                    var lev = levenshtein(path[i + x], contains[x]);
                    if (lev > MAX_LEV_DISTANCE) {
                        aborted = true;
                        break;
                    }
                    lev_total += lev;
                }
                if (!aborted) {
                    ret_lev = Math.min(ret_lev, Math.round(lev_total / clength));
                }
            }
            return ret_lev;
        }

        function typePassesFilter(filter, type) {
            // No filter
            if (filter <= NO_TYPE_FILTER) return true;

            // Exact match
            if (filter === type) return true;

            // Match related items
            var name = itemTypes[type];
            switch (itemTypes[filter]) {
                case "constant":
                    return name === "associatedconstant";
                case "fn":
                    return name === "method" || name === "tymethod";
                case "type":
                    return name === "primitive" || name === "associatedtype";
                case "trait":
                    return name === "traitalias";
            }

            // No match
            return false;
        }

        function createAliasFromItem(item) {
            return {
                crate: item.crate,
                name: item.name,
                path: item.path,
                desc: item.desc,
                ty: item.ty,
                parent: item.parent,
                type: item.type,
                is_alias: true,
            };
        }

        function handleAliases(ret, query, filterCrates) {
            // We separate aliases and crate aliases because we want to have current crate
            // aliases to be before the others in the displayed results.
            var aliases = [];
            var crateAliases = [];
            if (filterCrates !== undefined) {
                if (ALIASES[filterCrates] && ALIASES[filterCrates][query.search]) {
                    var query_aliases = ALIASES[filterCrates][query.search];
                    var len = query_aliases.length;
                    for (var i = 0; i < len; ++i) {
                        aliases.push(createAliasFromItem(searchIndex[query_aliases[i]]));
                    }
                }
            } else {
                Object.keys(ALIASES).forEach(function(crate) {
                    if (ALIASES[crate][query.search]) {
                        var pushTo = crate === window.currentCrate ? crateAliases : aliases;
                        var query_aliases = ALIASES[crate][query.search];
                        var len = query_aliases.length;
                        for (var i = 0; i < len; ++i) {
                            pushTo.push(createAliasFromItem(searchIndex[query_aliases[i]]));
                        }
                    }
                });
            }

            var sortFunc = function(aaa, bbb) {
                if (aaa.path < bbb.path) {
                    return 1;
                } else if (aaa.path === bbb.path) {
                    return 0;
                }
                return -1;
            };
            crateAliases.sort(sortFunc);
            aliases.sort(sortFunc);

            var pushFunc = function(alias) {
                alias.alias = query.raw;
                var res = buildHrefAndPath(alias);
                alias.displayPath = pathSplitter(res[0]);
                alias.fullPath = alias.displayPath + alias.name;
                alias.href = res[1];

                ret.others.unshift(alias);
                if (ret.others.length > MAX_RESULTS) {
                    ret.others.pop();
                }
            };
            onEach(aliases, pushFunc);
            onEach(crateAliases, pushFunc);
        }

        // quoted values mean literal search
        var nSearchWords = searchWords.length;
        var i, it;
        var ty;
        var fullId;
        var returned;
        var in_args;
        var len;
        if ((val.charAt(0) === "\"" || val.charAt(0) === "'") &&
            val.charAt(val.length - 1) === val.charAt(0))
        {
            val = extractGenerics(val.substr(1, val.length - 2));
            for (i = 0; i < nSearchWords; ++i) {
                if (filterCrates !== undefined && searchIndex[i].crate !== filterCrates) {
                    continue;
                }
                in_args = findArg(searchIndex[i], val, true, typeFilter);
                returned = checkReturned(searchIndex[i], val, true, typeFilter);
                ty = searchIndex[i];
                fullId = ty.id;

                if (searchWords[i] === val.name
                    && typePassesFilter(typeFilter, searchIndex[i].ty)
                    && results[fullId] === undefined) {
                    results[fullId] = {
                        id: i,
                        index: -1,
                        dontValidate: true,
                    };
                }
                if (in_args && results_in_args[fullId] === undefined) {
                    results_in_args[fullId] = {
                        id: i,
                        index: -1,
                        dontValidate: true,
                    };
                }
                if (returned && results_returned[fullId] === undefined) {
                    results_returned[fullId] = {
                        id: i,
                        index: -1,
                        dontValidate: true,
                    };
                }
            }
            query.inputs = [val];
            query.output = val;
            query.search = val;
        // searching by type
        } else if (val.search("->") > -1) {
            var trimmer = function(s) { return s.trim(); };
            var parts = val.split("->").map(trimmer);
            var input = parts[0];
            // sort inputs so that order does not matter
            var inputs = input.split(",").map(trimmer).sort();
            for (i = 0, len = inputs.length; i < len; ++i) {
                inputs[i] = extractGenerics(inputs[i]);
            }
            var output = extractGenerics(parts[1]);

            for (i = 0; i < nSearchWords; ++i) {
                if (filterCrates !== undefined && searchIndex[i].crate !== filterCrates) {
                    continue;
                }
                var type = searchIndex[i].type;
                ty = searchIndex[i];
                if (!type) {
                    continue;
                }
                fullId = ty.id;

                returned = checkReturned(ty, output, true, NO_TYPE_FILTER);
                if (output.name === "*" || returned) {
                    in_args = false;
                    var is_module = false;

                    if (input === "*") {
                        is_module = true;
                    } else {
                        var allFound = true;
                        for (it = 0, len = inputs.length; allFound && it < len; it++) {
                            allFound = checkType(type, inputs[it], true);
                        }
                        in_args = allFound;
                    }
                    if (in_args) {
                        results_in_args[fullId] = {
                            id: i,
                            index: -1,
                            dontValidate: true,
                        };
                    }
                    if (returned) {
                        results_returned[fullId] = {
                            id: i,
                            index: -1,
                            dontValidate: true,
                        };
                    }
                    if (is_module) {
                        results[fullId] = {
                            id: i,
                            index: -1,
                            dontValidate: true,
                        };
                    }
                }
            }
            query.inputs = inputs.map(function(input) {
                return input.name;
            });
            query.output = output.name;
        } else {
            query.inputs = [val];
            query.output = val;
            query.search = val;
            // gather matching search results up to a certain maximum
            val = val.replace(/_/g, "");

            var valGenerics = extractGenerics(val);

            var paths = valLower.split("::");
            removeEmptyStringsFromArray(paths);
            val = paths[paths.length - 1];
            var contains = paths.slice(0, paths.length > 1 ? paths.length - 1 : 1);

            var lev, j;
            for (j = 0; j < nSearchWords; ++j) {
                ty = searchIndex[j];
                if (!ty || (filterCrates !== undefined && ty.crate !== filterCrates)) {
                    continue;
                }
                var lev_add = 0;
                if (paths.length > 1) {
                    lev = checkPath(contains, paths[paths.length - 1], ty);
                    if (lev > MAX_LEV_DISTANCE) {
                        continue;
                    } else if (lev > 0) {
                        lev_add = lev / 10;
                    }
                }

                returned = MAX_LEV_DISTANCE + 1;
                in_args = MAX_LEV_DISTANCE + 1;
                var index = -1;
                // we want lev results to go lower than others
                lev = MAX_LEV_DISTANCE + 1;
                fullId = ty.id;

                if (searchWords[j].indexOf(split[i]) > -1 ||
                    searchWords[j].indexOf(val) > -1 ||
                    ty.normalizedName.indexOf(val) > -1)
                {
                    // filter type: ... queries
                    if (typePassesFilter(typeFilter, ty.ty) && results[fullId] === undefined) {
                        index = ty.normalizedName.indexOf(val);
                    }
                }
                if ((lev = levenshtein(searchWords[j], val)) <= MAX_LEV_DISTANCE) {
                    if (typePassesFilter(typeFilter, ty.ty)) {
                        lev += 1;
                    } else {
                        lev = MAX_LEV_DISTANCE + 1;
                    }
                }
                in_args = findArg(ty, valGenerics, false, typeFilter);
                returned = checkReturned(ty, valGenerics, false, typeFilter);

                lev += lev_add;
                if (lev > 0 && val.length > 3 && searchWords[j].indexOf(val) > -1) {
                    if (val.length < 6) {
                        lev -= 1;
                    } else {
                        lev = 0;
                    }
                }
                if (in_args <= MAX_LEV_DISTANCE) {
                    if (results_in_args[fullId] === undefined) {
                        results_in_args[fullId] = {
                            id: j,
                            index: index,
                            lev: in_args,
                        };
                    }
                    results_in_args[fullId].lev =
                        Math.min(results_in_args[fullId].lev, in_args);
                }
                if (returned <= MAX_LEV_DISTANCE) {
                    if (results_returned[fullId] === undefined) {
                        results_returned[fullId] = {
                            id: j,
                            index: index,
                            lev: returned,
                        };
                    }
                    results_returned[fullId].lev =
                        Math.min(results_returned[fullId].lev, returned);
                }
                if (typePassesFilter(typeFilter, ty.ty) &&
                        (index !== -1 || lev <= MAX_LEV_DISTANCE)) {
                    if (index !== -1 && paths.length < 2) {
                        lev = 0;
                    }
                    if (results[fullId] === undefined) {
                        results[fullId] = {
                            id: j,
                            index: index,
                            lev: lev,
                        };
                    }
                    results[fullId].lev = Math.min(results[fullId].lev, lev);
                }
            }
        }

        var ret = {
            "in_args": sortResults(results_in_args, true),
            "returned": sortResults(results_returned, true),
            "others": sortResults(results, false),
        };
        handleAliases(ret, query, filterCrates);
        return ret;
    }

    /**
     * Validate performs the following boolean logic. For example:
     * "File::open" will give IF A PARENT EXISTS => ("file" && "open")
     * exists in (name || path || parent) OR => ("file" && "open") exists in
     * (name || path )
     *
     * This could be written functionally, but I wanted to minimise
     * functions on stack.
     *
     * @param  {[string]} name   [The name of the result]
     * @param  {[string]} path   [The path of the result]
     * @param  {[string]} keys   [The keys to be used (["file", "open"])]
     * @param  {[object]} parent [The parent of the result]
     * @return {boolean}       [Whether the result is valid or not]
     */
    function validateResult(name, path, keys, parent) {
        for (var i = 0, len = keys.length; i < len; ++i) {
            // each check is for validation so we negate the conditions and invalidate
            if (!(
                // check for an exact name match
                name.indexOf(keys[i]) > -1 ||
                // then an exact path match
                path.indexOf(keys[i]) > -1 ||
                // next if there is a parent, check for exact parent match
                (parent !== undefined && parent.name !== undefined &&
                    parent.name.toLowerCase().indexOf(keys[i]) > -1) ||
                // lastly check to see if the name was a levenshtein match
                levenshtein(name, keys[i]) <= MAX_LEV_DISTANCE)) {
                return false;
            }
        }
        return true;
    }

    function getQuery(raw) {
        var matches, type, query;
        query = raw;

        matches = query.match(/^(fn|mod|struct|enum|trait|type|const|macro)\s*:\s*/i);
        if (matches) {
            type = matches[1].replace(/^const$/, "constant");
            query = query.substring(matches[0].length);
        }

        return {
            raw: raw,
            query: query,
            type: type,
            id: query + type
        };
    }

    function nextTab(direction) {
        var next = (searchState.currentTab + direction + 3) % searchState.focusedByTab.length;
        searchState.focusedByTab[searchState.currentTab] = document.activeElement;
        printTab(next);
        focusSearchResult();
    }

    // Focus the first search result on the active tab, or the result that
    // was focused last time this tab was active.
    function focusSearchResult() {
        var target = searchState.focusedByTab[searchState.currentTab] ||
            document.querySelectorAll(".search-results.active a").item(0) ||
            document.querySelectorAll("#titles > button").item(searchState.currentTab);
        if (target) {
            target.focus();
        }
    }

    function buildHrefAndPath(item) {
        var displayPath;
        var href;
        var type = itemTypes[item.ty];
        var name = item.name;
        var path = item.path;

        if (type === "mod") {
            displayPath = path + "::";
            href = window.rootPath + path.replace(/::/g, "/") + "/" +
                   name + "/index.html";
        } else if (type === "primitive" || type === "keyword") {
            displayPath = "";
            href = window.rootPath + path.replace(/::/g, "/") +
                   "/" + type + "." + name + ".html";
        } else if (type === "externcrate") {
            displayPath = "";
            href = window.rootPath + name + "/index.html";
        } else if (item.parent !== undefined) {
            var myparent = item.parent;
            var anchor = "#" + type + "." + name;
            var parentType = itemTypes[myparent.ty];
            var pageType = parentType;
            var pageName = myparent.name;

            if (parentType === "primitive") {
                displayPath = myparent.name + "::";
            } else if (type === "structfield" && parentType === "variant") {
                // Structfields belonging to variants are special: the
                // final path element is the enum name.
                var enumNameIdx = item.path.lastIndexOf("::");
                var enumName = item.path.substr(enumNameIdx + 2);
                path = item.path.substr(0, enumNameIdx);
                displayPath = path + "::" + enumName + "::" + myparent.name + "::";
                anchor = "#variant." + myparent.name + ".field." + name;
                pageType = "enum";
                pageName = enumName;
            } else {
                displayPath = path + "::" + myparent.name + "::";
            }
            href = window.rootPath + path.replace(/::/g, "/") +
                   "/" + pageType +
                   "." + pageName +
                   ".html" + anchor;
        } else {
            displayPath = item.path + "::";
            href = window.rootPath + item.path.replace(/::/g, "/") +
                   "/" + type + "." + name + ".html";
        }
        return [displayPath, href];
    }

    function escape(content) {
        var h1 = document.createElement("h1");
        h1.textContent = content;
        return h1.innerHTML;
    }

    function pathSplitter(path) {
        var tmp = "<span>" + path.replace(/::/g, "::</span><span>");
        if (tmp.endsWith("<span>")) {
            return tmp.slice(0, tmp.length - 6);
        }
        return tmp;
    }

    function addTab(array, query, display) {
        var extraClass = "";
        if (display === true) {
            extraClass = " active";
        }

        var output = document.createElement("div");
        var duplicates = {};
        var length = 0;
        if (array.length > 0) {
            output.className = "search-results " + extraClass;

            array.forEach(function(item) {
                if (item.is_alias !== true) {
                    if (duplicates[item.fullPath]) {
                        return;
                    }
                    duplicates[item.fullPath] = true;
                }

                var name = item.name;
                var type = itemTypes[item.ty];

                length += 1;

                var extra = "";
                if (type === "primitive") {
                    extra = " <i>(primitive type)</i>";
                } else if (type === "keyword") {
                    extra = " <i>(keyword)</i>";
                }

                var link = document.createElement("a");
                link.className = "result-" + type;
                link.href = item.href;

                var wrapper = document.createElement("div");
                var resultName = document.createElement("div");
                resultName.className = "result-name";

                if (item.is_alias) {
                    var alias = document.createElement("span");
                    alias.className = "alias";

                    var bold = document.createElement("b");
                    bold.innerText = item.alias;
                    alias.appendChild(bold);

                    alias.insertAdjacentHTML(
                        "beforeend",
                        "<span class=\"grey\"><i>&nbsp;- see&nbsp;</i></span>");

                    resultName.appendChild(alias);
                }
                resultName.insertAdjacentHTML(
                    "beforeend",
                    item.displayPath + "<span class=\"" + type + "\">" + name + extra + "</span>");
                wrapper.appendChild(resultName);

                var description = document.createElement("div");
                description.className = "desc";
                var spanDesc = document.createElement("span");
                spanDesc.insertAdjacentHTML("beforeend", item.desc);

                description.appendChild(spanDesc);
                wrapper.appendChild(description);
                link.appendChild(wrapper);
                output.appendChild(link);
            });
        } else {
            output.className = "search-failed" + extraClass;
            output.innerHTML = "No results :(<br/>" +
                "Try on <a href=\"https://duckduckgo.com/?q=" +
                encodeURIComponent("rust " + query.query) +
                "\">DuckDuckGo</a>?<br/><br/>" +
                "Or try looking in one of these:<ul><li>The <a " +
                "href=\"https://doc.rust-lang.org/reference/index.html\">Rust Reference</a> " +
                " for technical details about the language.</li><li><a " +
                "href=\"https://doc.rust-lang.org/rust-by-example/index.html\">Rust By " +
                "Example</a> for expository code examples.</a></li><li>The <a " +
                "href=\"https://doc.rust-lang.org/book/index.html\">Rust Book</a> for " +
                "introductions to language features and the language itself.</li><li><a " +
                "href=\"https://docs.rs\">Docs.rs</a> for documentation of crates released on" +
                " <a href=\"https://crates.io/\">crates.io</a>.</li></ul>";
        }
        return [output, length];
    }

    function makeTabHeader(tabNb, text, nbElems) {
        if (searchState.currentTab === tabNb) {
            return "<button class=\"selected\">" + text +
                   " <div class=\"count\">(" + nbElems + ")</div></button>";
        }
        return "<button>" + text + " <div class=\"count\">(" + nbElems + ")</div></button>";
    }

    function showResults(results, go_to_first) {
        var search = searchState.outputElement();
        if (go_to_first || (results.others.length === 1
            && getSettingValue("go-to-only-result") === "true"
            // By default, the search DOM element is "empty" (meaning it has no children not
            // text content). Once a search has been run, it won't be empty, even if you press
            // ESC or empty the search input (which also "cancels" the search).
            && (!search.firstChild || search.firstChild.innerText !== searchState.loadingText)))
        {
            var elem = document.createElement("a");
            elem.href = results.others[0].href;
            removeClass(elem, "active");
            // For firefox, we need the element to be in the DOM so it can be clicked.
            document.body.appendChild(elem);
            elem.click();
            return;
        }
        var query = getQuery(searchState.input.value);

        currentResults = query.id;

        var ret_others = addTab(results.others, query);
        var ret_in_args = addTab(results.in_args, query, false);
        var ret_returned = addTab(results.returned, query, false);

        // Navigate to the relevant tab if the current tab is empty, like in case users search
        // for "-> String". If they had selected another tab previously, they have to click on
        // it again.
        var currentTab = searchState.currentTab;
        if ((currentTab === 0 && ret_others[1] === 0) ||
                (currentTab === 1 && ret_in_args[1] === 0) ||
                (currentTab === 2 && ret_returned[1] === 0)) {
            if (ret_others[1] !== 0) {
                currentTab = 0;
            } else if (ret_in_args[1] !== 0) {
                currentTab = 1;
            } else if (ret_returned[1] !== 0) {
                currentTab = 2;
            }
        }

        var output = "<h1>Results for " + escape(query.query) +
            (query.type ? " (type: " + escape(query.type) + ")" : "") + "</h1>" +
            "<div id=\"titles\">" +
            makeTabHeader(0, "In Names", ret_others[1]) +
            makeTabHeader(1, "In Parameters", ret_in_args[1]) +
            makeTabHeader(2, "In Return Types", ret_returned[1]) +
            "</div>";

        var resultsElem = document.createElement("div");
        resultsElem.id = "results";
        resultsElem.appendChild(ret_others[0]);
        resultsElem.appendChild(ret_in_args[0]);
        resultsElem.appendChild(ret_returned[0]);

        search.innerHTML = output;
        search.appendChild(resultsElem);
        // Reset focused elements.
        searchState.focusedByTab = [null, null, null];
        searchState.showResults(search);
        var elems = document.getElementById("titles").childNodes;
        elems[0].onclick = function() { printTab(0); };
        elems[1].onclick = function() { printTab(1); };
        elems[2].onclick = function() { printTab(2); };
        printTab(currentTab);
    }

    function execSearch(query, searchWords, filterCrates) {
        function getSmallest(arrays, positions, notDuplicates) {
            var start = null;

            for (var it = 0, len = positions.length; it < len; ++it) {
                if (arrays[it].length > positions[it] &&
                    (start === null || start > arrays[it][positions[it]].lev) &&
                    !notDuplicates[arrays[it][positions[it]].fullPath]) {
                    start = arrays[it][positions[it]].lev;
                }
            }
            return start;
        }

        function mergeArrays(arrays) {
            var ret = [];
            var positions = [];
            var notDuplicates = {};

            for (var x = 0, arrays_len = arrays.length; x < arrays_len; ++x) {
                positions.push(0);
            }
            while (ret.length < MAX_RESULTS) {
                var smallest = getSmallest(arrays, positions, notDuplicates);

                if (smallest === null) {
                    break;
                }
                for (x = 0; x < arrays_len && ret.length < MAX_RESULTS; ++x) {
                    if (arrays[x].length > positions[x] &&
                            arrays[x][positions[x]].lev === smallest &&
                            !notDuplicates[arrays[x][positions[x]].fullPath]) {
                        ret.push(arrays[x][positions[x]]);
                        notDuplicates[arrays[x][positions[x]].fullPath] = true;
                        positions[x] += 1;
                    }
                }
            }
            return ret;
        }

        // Split search query by ",", while respecting angle bracket nesting.
        // Since "<" is an alias for the Ord family of traits, it also uses
        // lookahead to distinguish "<"-as-less-than from "<"-as-angle-bracket.
        //
        // tokenizeQuery("A<B, C>, D") == ["A<B, C>", "D"]
        // tokenizeQuery("A<B, C, D") == ["A<B", "C", "D"]
        function tokenizeQuery(raw) {
            var i, matched;
            var l = raw.length;
            var depth = 0;
            var nextAngle = /(<|>)/g;
            var ret = [];
            var start = 0;
            for (i = 0; i < l; ++i) {
                switch (raw[i]) {
                    case "<":
                        nextAngle.lastIndex = i + 1;
                        matched = nextAngle.exec(raw);
                        if (matched && matched[1] === '>') {
                            depth += 1;
                        }
                        break;
                    case ">":
                        if (depth > 0) {
                            depth -= 1;
                        }
                        break;
                    case ",":
                        if (depth === 0) {
                            ret.push(raw.substring(start, i));
                            start = i + 1;
                        }
                        break;
                }
            }
            if (start !== i) {
                ret.push(raw.substring(start, i));
            }
            return ret;
        }

        var queries = tokenizeQuery(query.raw);
        var results = {
            "in_args": [],
            "returned": [],
            "others": [],
        };

        for (var i = 0, len = queries.length; i < len; ++i) {
            query = queries[i].trim();
            if (query.length !== 0) {
                var tmp = execQuery(getQuery(query), searchWords, filterCrates);

                results.in_args.push(tmp.in_args);
                results.returned.push(tmp.returned);
                results.others.push(tmp.others);
            }
        }
        if (queries.length > 1) {
            return {
                "in_args": mergeArrays(results.in_args),
                "returned": mergeArrays(results.returned),
                "others": mergeArrays(results.others),
            };
        }
        return {
            "in_args": results.in_args[0],
            "returned": results.returned[0],
            "others": results.others[0],
        };
    }

    function getFilterCrates() {
        var elem = document.getElementById("crate-search");

        if (elem && elem.value !== "All crates" &&
            hasOwnPropertyRustdoc(rawSearchIndex, elem.value))
        {
            return elem.value;
        }
        return undefined;
    }

    function search(e, forced) {
        var params = searchState.getQueryStringParams();
        var query = getQuery(searchState.input.value.trim());

        if (e) {
            e.preventDefault();
        }

        if (query.query.length === 0) {
            return;
        }
        if (!forced && query.id === currentResults) {
            if (query.query.length > 0) {
                searchState.putBackSearch(searchState.input);
            }
            return;
        }

        // Update document title to maintain a meaningful browser history
        searchState.title = "Results for " + query.query + " - Rust";

        // Because searching is incremental by character, only the most
        // recent search query is added to the browser history.
        if (searchState.browserSupportsHistoryApi()) {
            var newURL = getNakedUrl() + "?search=" + encodeURIComponent(query.raw) +
                window.location.hash;
            if (!history.state && !params.search) {
                history.pushState(query, "", newURL);
            } else {
                history.replaceState(query, "", newURL);
            }
        }

        var filterCrates = getFilterCrates();
        showResults(execSearch(query, index, filterCrates), params.go_to_first);
    }

    function buildIndex(rawSearchIndex) {
        searchIndex = [];
        var searchWords = [];
        var i, word;
        var currentIndex = 0;
        var id = 0;

        for (var crate in rawSearchIndex) {
            if (!hasOwnPropertyRustdoc(rawSearchIndex, crate)) {
                continue;
            }

            var crateSize = 0;

            searchWords.push(crate);
            // This object should have exactly the same set of fields as the "row"
            // object defined below. Your JavaScript runtime will thank you.
            // https://mathiasbynens.be/notes/shapes-ics
            var crateRow = {
                crate: crate,
                ty: 1, // == ExternCrate
                name: crate,
                path: "",
                desc: rawSearchIndex[crate].doc,
                parent: undefined,
                type: null,
                id: id,
                normalizedName: crate.indexOf("_") === -1 ? crate : crate.replace(/_/g, ""),
            };
            id += 1;
            searchIndex.push(crateRow);
            currentIndex += 1;

            // an array of (Number) item types
            var itemTypes = rawSearchIndex[crate].t;
            // an array of (String) item names
            var itemNames = rawSearchIndex[crate].n;
            // an array of (String) full paths (or empty string for previous path)
            var itemPaths = rawSearchIndex[crate].q;
            // an array of (String) descriptions
            var itemDescs = rawSearchIndex[crate].d;
            // an array of (Number) the parent path index + 1 to `paths`, or 0 if none
            var itemParentIdxs = rawSearchIndex[crate].i;
            // an array of (Object | null) the type of the function, if any
            var itemFunctionSearchTypes = rawSearchIndex[crate].f;
            // an array of [(Number) item type,
            //              (String) name]
            var paths = rawSearchIndex[crate].p;
            // an array of [(String) alias name
            //             [Number] index to items]
            var aliases = rawSearchIndex[crate].a;

            // convert `rawPaths` entries into object form
            var len = paths.length;
            for (i = 0; i < len; ++i) {
                paths[i] = {ty: paths[i][0], name: paths[i][1]};
            }

            // convert `item*` into an object form, and construct word indices.
            //
            // before any analysis is performed lets gather the search terms to
            // search against apart from the rest of the data.  This is a quick
            // operation that is cached for the life of the page state so that
            // all other search operations have access to this cached data for
            // faster analysis operations
            len = itemTypes.length;
            var lastPath = "";
            for (i = 0; i < len; ++i) {
                // This object should have exactly the same set of fields as the "crateRow"
                // object defined above.
                if (typeof itemNames[i] === "string") {
                    word = itemNames[i].toLowerCase();
                    searchWords.push(word);
                } else {
                    word = "";
                    searchWords.push("");
                }
                var row = {
                    crate: crate,
                    ty: itemTypes[i],
                    name: itemNames[i],
                    path: itemPaths[i] ? itemPaths[i] : lastPath,
                    desc: itemDescs[i],
                    parent: itemParentIdxs[i] > 0 ? paths[itemParentIdxs[i] - 1] : undefined,
                    type: itemFunctionSearchTypes[i],
                    id: id,
                    normalizedName: word.indexOf("_") === -1 ? word : word.replace(/_/g, ""),
                };
                id += 1;
                searchIndex.push(row);
                lastPath = row.path;
                crateSize += 1;
            }

            if (aliases) {
                ALIASES[crate] = {};
                var j, local_aliases;
                for (var alias_name in aliases) {
                    if (!hasOwnPropertyRustdoc(aliases, alias_name)) {
                        continue;
                    }

                    if (!hasOwnPropertyRustdoc(ALIASES[crate], alias_name)) {
                        ALIASES[crate][alias_name] = [];
                    }
                    local_aliases = aliases[alias_name];
                    for (j = 0, len = local_aliases.length; j < len; ++j) {
                        ALIASES[crate][alias_name].push(local_aliases[j] + currentIndex);
                    }
                }
            }
            currentIndex += crateSize;
        }
        return searchWords;
    }

    function registerSearchEvents() {
        var searchAfter500ms = function() {
            searchState.clearInputTimeout();
            if (searchState.input.value.length === 0) {
                if (searchState.browserSupportsHistoryApi()) {
                    history.replaceState("", window.currentCrate + " - Rust",
                        getNakedUrl() + window.location.hash);
                }
                searchState.hideResults();
            } else {
                searchState.timeout = setTimeout(search, 500);
            }
        };
        searchState.input.onkeyup = searchAfter500ms;
        searchState.input.oninput = searchAfter500ms;
        document.getElementsByClassName("search-form")[0].onsubmit = function(e) {
            e.preventDefault();
            searchState.clearInputTimeout();
            search();
        };
        searchState.input.onchange = function(e) {
            if (e.target !== document.activeElement) {
                // To prevent doing anything when it's from a blur event.
                return;
            }
            // Do NOT e.preventDefault() here. It will prevent pasting.
            searchState.clearInputTimeout();
            // zero-timeout necessary here because at the time of event handler execution the
            // pasted content is not in the input field yet. Shouldn’t make any difference for
            // change, though.
            setTimeout(search, 0);
        };
        searchState.input.onpaste = searchState.input.onchange;

        searchState.outputElement().addEventListener("keydown", function(e) {
            // We only handle unmodified keystrokes here. We don't want to interfere with,
            // for instance, alt-left and alt-right for history navigation.
            if (e.altKey || e.ctrlKey || e.shiftKey || e.metaKey) {
                return;
            }
            // up and down arrow select next/previous search result, or the
            // search box if we're already at the top.
            if (e.which === 38) { // up
                var previous = document.activeElement.previousElementSibling;
                if (previous) {
                    previous.focus();
                } else {
                    searchState.focus();
                }
                e.preventDefault();
            } else if (e.which === 40) { // down
                var next = document.activeElement.nextElementSibling;
                if (next) {
                    next.focus();
                }
                var rect = document.activeElement.getBoundingClientRect();
                if (window.innerHeight - rect.bottom < rect.height) {
                    window.scrollBy(0, rect.height);
                }
                e.preventDefault();
            } else if (e.which === 37) { // left
                nextTab(-1);
                e.preventDefault();
            } else if (e.which === 39) { // right
                nextTab(1);
                e.preventDefault();
            }
        });

        searchState.input.addEventListener("keydown", function(e) {
            if (e.which === 40) { // down
                focusSearchResult();
                e.preventDefault();
            }
        });


        var selectCrate = document.getElementById("crate-search");
        if (selectCrate) {
            selectCrate.onchange = function() {
                updateLocalStorage("rustdoc-saved-filter-crate", selectCrate.value);
                // In case you "cut" the entry from the search input, then change the crate filter
                // before paste back the previous search, you get the old search results without
                // the filter. To prevent this, we need to remove the previous results.
                currentResults = null;
                search(undefined, true);
            };
        }

        // Push and pop states are used to add search results to the browser
        // history.
        if (searchState.browserSupportsHistoryApi()) {
            // Store the previous <title> so we can revert back to it later.
            var previousTitle = document.title;

            window.addEventListener("popstate", function(e) {
                var params = searchState.getQueryStringParams();
                // Revert to the previous title manually since the History
                // API ignores the title parameter.
                document.title = previousTitle;
                // When browsing forward to search results the previous
                // search will be repeated, so the currentResults are
                // cleared to ensure the search is successful.
                currentResults = null;
                // Synchronize search bar with query string state and
                // perform the search. This will empty the bar if there's
                // nothing there, which lets you really go back to a
                // previous state with nothing in the bar.
                if (params.search && params.search.length > 0) {
                    searchState.input.value = params.search;
                    // Some browsers fire "onpopstate" for every page load
                    // (Chrome), while others fire the event only when actually
                    // popping a state (Firefox), which is why search() is
                    // called both here and at the end of the startSearch()
                    // function.
                    search(e);
                } else {
                    searchState.input.value = "";
                    // When browsing back from search results the main page
                    // visibility must be reset.
                    searchState.hideResults();
                }
            });
        }

        // This is required in firefox to avoid this problem: Navigating to a search result
        // with the keyboard, hitting enter, and then hitting back would take you back to
        // the doc page, rather than the search that should overlay it.
        // This was an interaction between the back-forward cache and our handlers
        // that try to sync state between the URL and the search input. To work around it,
        // do a small amount of re-init on page show.
        window.onpageshow = function(){
            var qSearch = searchState.getQueryStringParams().search;
            if (searchState.input.value === "" && qSearch) {
                searchState.input.value = qSearch;
            }
            search();
        };
    }

    index = buildIndex(rawSearchIndex);
    registerSearchEvents();
    // If there's a search term in the URL, execute the search now.
    if (searchState.getQueryStringParams().search) {
        search();
    }
};

if (window.searchIndex !== undefined) {
    initSearch(window.searchIndex);
}

})();
