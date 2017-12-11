/*!
 * Copyright 2014 The Rust Project Developers. See the COPYRIGHT
 * file at the top-level directory of this distribution and at
 * http://rust-lang.org/COPYRIGHT.
 *
 * Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
 * http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
 * <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
 * option. This file may not be copied, modified, or distributed
 * except according to those terms.
 */

/*jslint browser: true, es5: true */
/*globals $: true, rootPath: true */

(function() {
    "use strict";

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
                     "foreigntype"];

    // On the search screen, so you remain on the last tab you opened.
    //
    // 0 for "Types/modules"
    // 1 for "As parameters"
    // 2 for "As return value"
    var currentTab = 0;

    function hasClass(elem, className) {
        if (elem && className && elem.className) {
            var elemClass = elem.className;
            var start = elemClass.indexOf(className);
            if (start === -1) {
                return false;
            } else if (elemClass.length === className.length) {
                return true;
            } else {
                if (start > 0 && elemClass[start - 1] !== ' ') {
                    return false;
                }
                var end = start + className.length;
                if (end < elemClass.length && elemClass[end] !== ' ') {
                    return false;
                }
                return true;
            }
            if (start > 0 && elemClass[start - 1] !== ' ') {
                return false;
            }
            var end = start + className.length;
            if (end < elemClass.length && elemClass[end] !== ' ') {
                return false;
            }
            return true;
        }
        return false;
    }

    function addClass(elem, className) {
        if (elem && className && !hasClass(elem, className)) {
            if (elem.className && elem.className.length > 0) {
                elem.className += ' ' + className;
            } else {
                elem.className = className;
            }
        }
    }

    function removeClass(elem, className) {
        if (elem && className && elem.className) {
            elem.className = (" " + elem.className + " ").replace(" " + className + " ", " ")
                                                         .trim();
        }
    }

    function onEach(arr, func) {
        if (arr && arr.length > 0 && func) {
            for (var i = 0; i < arr.length; i++) {
                func(arr[i]);
            }
        }
    }

    function isHidden(elem) {
        return (elem.offsetParent === null)
    }

    // used for special search precedence
    var TY_PRIMITIVE = itemTypes.indexOf("primitive");

    onEach(document.getElementsByClassName('js-only'), function(e) {
        removeClass(e, 'js-only');
    });

    function getQueryStringParams() {
        var params = {};
        window.location.search.substring(1).split("&").
            map(function(s) {
                var pair = s.split("=");
                params[decodeURIComponent(pair[0])] =
                    typeof pair[1] === "undefined" ?
                            null : decodeURIComponent(pair[1]);
            });
        return params;
    }

    function browserSupportsHistoryApi() {
        return document.location.protocol != "file:" &&
          window.history && typeof window.history.pushState === "function";
    }

    function highlightSourceLines(ev) {
        var search = document.getElementById("search");
        var i, from, to, match = window.location.hash.match(/^#?(\d+)(?:-(\d+))?$/);
        if (match) {
            from = parseInt(match[1], 10);
            to = Math.min(50000, parseInt(match[2] || match[1], 10));
            from = Math.min(from, to);
            var elem = document.getElementById(from);
            if (!elem) {
                return;
            }
            if (ev === null) {
                var x = document.getElementById(from);
                if (x) {
                    x.scrollIntoView();
                }
            }
            onEach(document.getElementsByClassName('line-numbers'), function(e) {
                onEach(e.getElementsByTagName('span'), function(i_e) {
                    removeClass(i_e, 'line-highlighted');
                });
            })
            for (i = from; i <= to; ++i) {
                addClass(document.getElementById(i), 'line-highlighted');
            }
        } else if (ev !== null && search && !hasClass(search, "hidden") && ev.newURL) {
            addClass(search, "hidden");
            removeClass(document.getElementById("main"), "hidden");
            var hash = ev.newURL.slice(ev.newURL.indexOf('#') + 1);
            if (browserSupportsHistoryApi()) {
                history.replaceState(hash, "", "?search=#" + hash);
            }
            var elem = document.getElementById(hash);
            if (elem) {
                elem.scrollIntoView();
            }
        }
    }
    highlightSourceLines(null);
    window.onhashchange = highlightSourceLines;

    // Gets the human-readable string for the virtual-key code of the
    // given KeyboardEvent, ev.
    //
    // This function is meant as a polyfill for KeyboardEvent#key,
    // since it is not supported in Trident.  We also test for
    // KeyboardEvent#keyCode because the handleShortcut handler is
    // also registered for the keydown event, because Blink doesn't fire
    // keypress on hitting the Escape key.
    //
    // So I guess you could say things are getting pretty interoperable.
    function getVirtualKey(ev) {
        if ("key" in ev && typeof ev.key != "undefined")
            return ev.key;

        var c = ev.charCode || ev.keyCode;
        if (c == 27)
            return "Escape";
        return String.fromCharCode(c);
    }

    function displayHelp(display, ev) {
        if (display === true) {
            if (hasClass(help, "hidden")) {
                ev.preventDefault();
                removeClass(help, "hidden");
                addClass(document.body, "blur");
            }
        } else if (!hasClass(help, "hidden")) {
            ev.preventDefault();
            addClass(help, "hidden");
            removeClass(document.body, "blur");
        }
    }

    function handleShortcut(ev) {
        if (document.activeElement.tagName === "INPUT")
            return;

        // Don't interfere with browser shortcuts
        if (ev.ctrlKey || ev.altKey || ev.metaKey)
            return;

        var help = document.getElementById("help");
        switch (getVirtualKey(ev)) {
        case "Escape":
            var search = document.getElementById("search");
            if (!hasClass(help, "hidden")) {
                displayHelp(false, ev);
            } else if (!hasClass(search, "hidden")) {
                ev.preventDefault();
                addClass(search, "hidden");
                removeClass(document.getElementById("main"), "hidden");
            }
            break;

        case "s":
        case "S":
            displayHelp(false, ev);
            ev.preventDefault();
            focusSearchBar();
            break;

        case "+":
        case "-":
            ev.preventDefault();
            toggleAllDocs();
            break;

        case "?":
            if (ev.shiftKey) {
                displayHelp(true, ev);
            }
            break;
        }
    }

    document.onkeypress = handleShortcut;
    document.onkeydown = handleShortcut;
    document.onclick = function(ev) {
        if (hasClass(ev.target, 'collapse-toggle')) {
            collapseDocs(ev.target);
        } else if (hasClass(ev.target.parentNode, 'collapse-toggle')) {
            collapseDocs(ev.target.parentNode);
        } else if (ev.target.tagName === 'SPAN' && hasClass(ev.target.parentNode, 'line-numbers')) {
            var prev_id = 0;

            var set_fragment = function (name) {
                if (browserSupportsHistoryApi()) {
                    history.replaceState(null, null, '#' + name);
                    window.hashchange();
                } else {
                    location.replace('#' + name);
                }
            };

            var cur_id = parseInt(ev.target.id, 10);

            if (ev.shiftKey && prev_id) {
                if (prev_id > cur_id) {
                    var tmp = prev_id;
                    prev_id = cur_id;
                    cur_id = tmp;
                }

                set_fragment(prev_id + '-' + cur_id);
            } else {
                prev_id = cur_id;

                set_fragment(cur_id);
            }
        } else if (!hasClass(document.getElementById("help"), "hidden")) {
            addClass(document.getElementById("help"), "hidden");
            removeClass(document.body, "blur");
        }
    };

    var x = document.getElementsByClassName('version-selector');
    if (x.length > 0) {
        x[0].onchange = function() {
            var i, match,
                url = document.location.href,
                stripped = '',
                len = rootPath.match(/\.\.\//g).length + 1;

            for (i = 0; i < len; ++i) {
                match = url.match(/\/[^\/]*$/);
                if (i < len - 1) {
                    stripped = match[0] + stripped;
                }
                url = url.substring(0, url.length - match[0].length);
            }

            url += '/' + document.getElementsByClassName('version-selector')[0].value + stripped;

            document.location.href = url;
        };
    }

    /**
     * A function to compute the Levenshtein distance between two strings
     * Licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported
     * Full License can be found at http://creativecommons.org/licenses/by-sa/3.0/legalcode
     * This code is an unmodified version of the code written by Marco de Wit
     * and was found at http://stackoverflow.com/a/18514751/745719
     */
    var levenshtein = (function() {
        var row2 = [];
        return function(s1, s2) {
            if (s1 === s2) {
                return 0;
            }
            var s1_len = s1.length, s2_len = s2.length;
            if (s1_len && s2_len) {
                var i1 = 0, i2 = 0, a, b, c, c2, row = row2;
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
        };
    })();

    function initSearch(rawSearchIndex) {
        var currentResults, index, searchIndex;
        var MAX_LEV_DISTANCE = 3;
        var MAX_RESULTS = 200;
        var params = getQueryStringParams();

        // Populate search bar with query string search term when provided,
        // but only if the input bar is empty. This avoid the obnoxious issue
        // where you start trying to do a search, and the index loads, and
        // suddenly your search is gone!
        if (document.getElementsByClassName("search-input")[0].value === "") {
            document.getElementsByClassName("search-input")[0].value = params.search || '';
        }

        /**
         * Executes the query and builds an index of results
         * @param  {[Object]} query     [The user query]
         * @param  {[type]} max         [The maximum results returned]
         * @param  {[type]} searchWords [The list of search words to query
         *                               against]
         * @return {[type]}             [A search index of results]
         */
        function execQuery(query, max, searchWords) {
            var valLower = query.query.toLowerCase(),
                val = valLower,
                typeFilter = itemTypeFromName(query.type),
                results = {}, results_in_args = {}, results_returned = {},
                split = valLower.split("::");

            for (var z = 0; z < split.length; ++z) {
                if (split[z] === "") {
                    split.splice(z, 1);
                    z -= 1;
                }
            }

            function transformResults(results, isType) {
                var out = [];
                for (i = 0; i < results.length; ++i) {
                    if (results[i].id > -1) {
                        var obj = searchIndex[results[i].id];
                        obj.lev = results[i].lev;
                        if (isType !== true || obj.type) {
                            out.push(obj);
                        }
                    }
                    if (out.length >= MAX_RESULTS) {
                        break;
                    }
                }
                return out;
            }

            function sortResults(results, isType) {
                var ar = [];
                for (var entry in results) {
                    if (results.hasOwnProperty(entry)) {
                        ar.push(results[entry]);
                    }
                }
                results = ar;
                var nresults = results.length;
                for (var i = 0; i < nresults; ++i) {
                    results[i].word = searchWords[results[i].id];
                    results[i].item = searchIndex[results[i].id] || {};
                }
                // if there are no results then return to default and fail
                if (results.length === 0) {
                    return [];
                }

                results.sort(function(aaa, bbb) {
                    var a, b;

                    // Sort by non levenshtein results and then levenshtein results by the distance
                    // (less changes required to match means higher rankings)
                    a = (aaa.lev);
                    b = (bbb.lev);
                    if (a !== b) { return a - b; }

                    // sort by crate (non-current crate goes later)
                    a = (aaa.item.crate !== window.currentCrate);
                    b = (bbb.item.crate !== window.currentCrate);
                    if (a !== b) { return a - b; }

                    // sort by exact match (mismatch goes later)
                    a = (aaa.word !== valLower);
                    b = (bbb.word !== valLower);
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

                    // special precedence for primitive pages
                    if ((aaa.item.ty === TY_PRIMITIVE) && (bbb.item.ty !== TY_PRIMITIVE)) {
                        return -1;
                    }
                    if ((bbb.item.ty === TY_PRIMITIVE) && (aaa.item.ty !== TY_PRIMITIVE)) {
                        return 1;
                    }

                    // sort by description (no description goes later)
                    a = (aaa.item.desc === '');
                    b = (bbb.item.desc === '');
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

                for (var i = 0; i < results.length; ++i) {
                    var result = results[i];

                    // this validation does not make sense when searching by types
                    if (result.dontValidate) {
                        continue;
                    }
                    var name = result.item.name.toLowerCase(),
                        path = result.item.path.toLowerCase(),
                        parent = result.item.parent;

                    if (isType !== true &&
                        validateResult(name, path, split, parent) === false)
                    {
                        result.id = -1;
                    }
                }
                return transformResults(results);
            }

            function extractGenerics(val) {
                val = val.toLowerCase();
                if (val.indexOf('<') !== -1) {
                    var values = val.substring(val.indexOf('<') + 1, val.lastIndexOf('>'));
                    return {
                        name: val.substring(0, val.indexOf('<')),
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
                var lev_distance = MAX_LEV_DISTANCE + 1;
                if (val.generics.length > 0) {
                    if (obj.generics && obj.generics.length >= val.generics.length) {
                        var elems = obj.generics.slice(0);
                        var total = 0;
                        var done = 0;
                        // We need to find the type that matches the most to remove it in order
                        // to move forward.
                        for (var y = 0; y < val.generics.length; ++y) {
                            var lev = { pos: -1, lev: MAX_LEV_DISTANCE + 1};
                            for (var x = 0; x < elems.length; ++x) {
                                var tmp_lev = levenshtein(elems[x], val.generics[y]);
                                if (tmp_lev < lev.lev) {
                                    lev.lev = tmp_lev;
                                    lev.pos = x;
                                }
                            }
                            if (lev.pos !== -1) {
                                elems.splice(lev.pos, 1);
                                lev_distance = Math.min(lev.lev, lev_distance);
                                total += lev.lev;
                                done += 1;
                            } else {
                                return MAX_LEV_DISTANCE + 1;
                            }
                        }
                        return lev_distance;//Math.ceil(total / done);
                    }
                }
                return MAX_LEV_DISTANCE + 1;
            }

            // Check for type name and type generics (if any).
            function checkType(obj, val, literalSearch) {
                var lev_distance = MAX_LEV_DISTANCE + 1;
                if (obj.name === val.name) {
                    if (literalSearch === true) {
                        if (val.generics.length !== 0) {
                            if (obj.generics && obj.length >= val.generics.length) {
                                var elems = obj.generics.slice(0);
                                var allFound = true;
                                var x;

                                for (var y = 0; allFound === true && y < val.generics.length; ++y) {
                                    allFound = false;
                                    for (x = 0; allFound === false && x < elems.length; ++x) {
                                        allFound = elems[x] === val.generics[y];
                                    }
                                    if (allFound === true) {
                                        elems.splice(x - 1, 1);
                                    }
                                }
                                if (allFound === true) {
                                    return true;
                                }
                            } else {
                                return false;
                            }
                        }
                        return true;
                    }
                    // If the type has generics but don't match, then it won't return at this point.
                    // Otherwise, `checkGenerics` will return 0 and it'll return.
                    if (obj.generics && obj.generics.length !== 0) {
                        var tmp_lev = checkGenerics(obj, val);
                        if (tmp_lev <= MAX_LEV_DISTANCE) {
                            return tmp_lev;
                        }
                    } else {
                        return 0;
                    }
                }
                // Names didn't match so let's check if one of the generic types could.
                if (literalSearch === true) {
                     if (obj.generics.length > 0) {
                        for (var x = 0; x < obj.generics.length; ++x) {
                            if (obj.generics[x] === val.name) {
                                return true;
                            }
                        }
                    }
                    return false;
                }
                var lev_distance = Math.min(levenshtein(obj.name, val.name), lev_distance);
                if (lev_distance <= MAX_LEV_DISTANCE) {
                    lev_distance = Math.min(checkGenerics(obj, val), lev_distance);
                } else if (obj.generics && obj.generics.length > 0) {
                    // We can check if the type we're looking for is inside the generics!
                    for (var x = 0; x < obj.generics.length; ++x) {
                        lev_distance = Math.min(levenshtein(obj.generics[x], val.name),
                                                lev_distance);
                    }
                }
                // Now whatever happens, the returned distance is "less good" so we should mark it
                // as such, and so we add 1 to the distance to make it "less good".
                return lev_distance + 1;
            }

            function findArg(obj, val, literalSearch) {
                var lev_distance = MAX_LEV_DISTANCE + 1;

                if (obj && obj.type && obj.type.inputs.length > 0) {
                    for (var i = 0; i < obj.type.inputs.length; i++) {
                        var tmp = checkType(obj.type.inputs[i], val, literalSearch);
                        if (literalSearch === true && tmp === true) {
                            return true;
                        }
                        lev_distance = Math.min(tmp, lev_distance);
                        if (lev_distance === 0) {
                            return 0;
                        }
                    }
                }
                return literalSearch === true ? false : lev_distance;
            }

            function checkReturned(obj, val, literalSearch) {
                var lev_distance = MAX_LEV_DISTANCE + 1;

                if (obj && obj.type && obj.type.output) {
                    var tmp = checkType(obj.type.output, val, literalSearch);
                    if (literalSearch === true && tmp === true) {
                        return true;
                    }
                    lev_distance = Math.min(tmp, lev_distance);
                    if (lev_distance === 0) {
                        return 0;
                    }
                }
                return literalSearch === true ? false : lev_distance;
            }

            function checkPath(startsWith, lastElem, ty) {
                var ret_lev = MAX_LEV_DISTANCE + 1;
                var path = ty.path.split("::");

                if (ty.parent && ty.parent.name) {
                    path.push(ty.parent.name.toLowerCase());
                }

                if (startsWith.length > path.length) {
                    return MAX_LEV_DISTANCE + 1;
                }
                for (var i = 0; i < path.length; ++i) {
                    if (i + startsWith.length > path.length) {
                        break;
                    }
                    var lev_total = 0;
                    var aborted = false;
                    for (var x = 0; x < startsWith.length; ++x) {
                        var lev = levenshtein(path[i + x], startsWith[x]);
                        if (lev > MAX_LEV_DISTANCE) {
                            aborted = true;
                            break;
                        }
                        lev_total += lev;
                    }
                    if (aborted === false) {
                        var extra = MAX_LEV_DISTANCE + 1;
                        if (i + startsWith.length < path.length) {
                            extra = levenshtein(path[i + startsWith.length], lastElem);
                        }
                        if (extra > MAX_LEV_DISTANCE) {
                            extra = levenshtein(ty.name, lastElem);
                        }
                        if (extra < MAX_LEV_DISTANCE + 1) {
                            lev_total += extra;
                            ret_lev = Math.min(ret_lev,
                                               Math.round(lev_total / (startsWith.length + 1)));
                        }
                    }
                }
                return ret_lev;
            }

            function typePassesFilter(filter, type) {
                // No filter
                if (filter < 0) return true;

                // Exact match
                if (filter === type) return true;

                // Match related items
                var name = itemTypes[type];
                switch (itemTypes[filter]) {
                    case "constant":
                        return (name == "associatedconstant");
                    case "fn":
                        return (name == "method" || name == "tymethod");
                    case "type":
                        return (name == "primitive");
                }

                // No match
                return false;
            }

            function generateId(ty) {
                if (ty.parent && ty.parent.name) {
                    return itemTypes[ty.ty] + ty.path + ty.parent.name + ty.name;
                }
                return itemTypes[ty.ty] + ty.path + ty.name;
            }

            // quoted values mean literal search
            var nSearchWords = searchWords.length;
            if ((val.charAt(0) === "\"" || val.charAt(0) === "'") &&
                val.charAt(val.length - 1) === val.charAt(0))
            {
                val = extractGenerics(val.substr(1, val.length - 2));
                for (var i = 0; i < nSearchWords; ++i) {
                    var in_args = findArg(searchIndex[i], val, true);
                    var returned = checkReturned(searchIndex[i], val, true);
                    var ty = searchIndex[i];
                    var fullId = generateId(ty);

                    if (searchWords[i] === val.name) {
                        // filter type: ... queries
                        if (typePassesFilter(typeFilter, searchIndex[i].ty) &&
                            results[fullId] === undefined)
                        {
                            results[fullId] = {id: i, index: -1};
                        }
                    } else if ((in_args === true || returned === true) &&
                               typePassesFilter(typeFilter, searchIndex[i].ty)) {
                        if (in_args === true || returned === true) {
                            if (in_args === true) {
                                results_in_args[fullId] = {
                                    id: i,
                                    index: -1,
                                    dontValidate: true,
                                };
                            }
                            if (returned === true) {
                                results_returned[fullId] = {
                                    id: i,
                                    index: -1,
                                    dontValidate: true,
                                };
                            }
                        } else {
                            results[fullId] = {
                                id: i,
                                index: -1,
                                dontValidate: true,
                            };
                        }
                    }
                }
                query.inputs = [val];
                query.output = val;
                query.search = val;
            // searching by type
            } else if (val.search("->") > -1) {
                var trimmer = function (s) { return s.trim(); };
                var parts = val.split("->").map(trimmer);
                var input = parts[0];
                // sort inputs so that order does not matter
                var inputs = input.split(",").map(trimmer).sort();
                for (var i = 0; i < inputs.length; ++i) {
                    inputs[i] = extractGenerics(inputs[i]);
                }
                var output = extractGenerics(parts[1]);

                for (var i = 0; i < nSearchWords; ++i) {
                    var type = searchIndex[i].type;
                    var ty = searchIndex[i];
                    if (!type) {
                        continue;
                    }
                    var fullId = generateId(ty);

                    // allow searching for void (no output) functions as well
                    var typeOutput = type.output ? type.output.name : "";
                    var returned = checkReturned(ty, output, true);
                    if (output.name === "*" || returned === true) {
                        var in_args = false;
                        var module = false;

                        if (input === "*") {
                            module = true;
                        } else {
                            var allFound = true;
                            for (var it = 0; allFound === true && it < inputs.length; it++) {
                                allFound = checkType(type, inputs[it], true);
                            }
                            in_args = allFound;
                        }
                        if (in_args === true) {
                            results_in_args[fullId] = {
                                id: i,
                                index: -1,
                                dontValidate: true,
                            };
                        }
                        if (returned === true) {
                            results_returned[fullId] = {
                                id: i,
                                index: -1,
                                dontValidate: true,
                            };
                        }
                        if (module === true) {
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
                val = val.replace(/\_/g, "");

                var valGenerics = extractGenerics(val);

                var paths = valLower.split("::");
                var j;
                for (j = 0; j < paths.length; ++j) {
                    if (paths[j] === "") {
                        paths.splice(j, 1);
                        j -= 1;
                    }
                }
                val = paths[paths.length - 1];
                var startsWith = paths.slice(0, paths.length > 1 ? paths.length - 1 : 1);

                for (j = 0; j < nSearchWords; ++j) {
                    var lev_distance;
                    var ty = searchIndex[j];
                    if (!ty) {
                        continue;
                    }
                    var lev_add = 0;
                    if (paths.length > 1) {
                        var lev = checkPath(startsWith, paths[paths.length - 1], ty);
                        if (lev > MAX_LEV_DISTANCE) {
                            continue;
                        } else if (lev > 0) {
                            lev_add = 1;
                        }
                    }

                    var returned = MAX_LEV_DISTANCE + 1;
                    var in_args = MAX_LEV_DISTANCE + 1;
                    var index = -1;
                    // we want lev results to go lower than others
                    var lev = MAX_LEV_DISTANCE + 1;
                    var fullId = generateId(ty);

                    if (searchWords[j].indexOf(split[i]) > -1 ||
                        searchWords[j].indexOf(val) > -1 ||
                        searchWords[j].replace(/_/g, "").indexOf(val) > -1)
                    {
                        // filter type: ... queries
                        if (typePassesFilter(typeFilter, ty.ty) && results[fullId] === undefined) {
                            index = searchWords[j].replace(/_/g, "").indexOf(val);
                        }
                    }
                    if ((lev = levenshtein(searchWords[j], val)) <= MAX_LEV_DISTANCE) {
                        if (typePassesFilter(typeFilter, ty.ty) === false) {
                            lev = MAX_LEV_DISTANCE + 1;
                        } else {
                            lev += 1;
                        }
                    }
                    if ((in_args = findArg(ty, valGenerics)) <= MAX_LEV_DISTANCE) {
                        if (typePassesFilter(typeFilter, ty.ty) === false) {
                            in_args = MAX_LEV_DISTANCE + 1;
                        }
                    }
                    if ((returned = checkReturned(ty, valGenerics)) <= MAX_LEV_DISTANCE) {
                        if (typePassesFilter(typeFilter, ty.ty) === false) {
                            returned = MAX_LEV_DISTANCE + 1;
                        }
                    }

                    lev += lev_add;
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
                    if (index !== -1 || lev <= MAX_LEV_DISTANCE) {
                        if (index !== -1) {
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

            return {
                'in_args': sortResults(results_in_args, true),
                'returned': sortResults(results_returned, true),
                'others': sortResults(results),
            };
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
         * @return {[boolean]}       [Whether the result is valid or not]
         */
        function validateResult(name, path, keys, parent) {
            for (var i = 0; i < keys.length; ++i) {
                // each check is for validation so we negate the conditions and invalidate
                if (!(
                    // check for an exact name match
                    name.indexOf(keys[i]) > -1 ||
                    // then an exact path match
                    path.indexOf(keys[i]) > -1 ||
                    // next if there is a parent, check for exact parent match
                    (parent !== undefined &&
                        parent.name.toLowerCase().indexOf(keys[i]) > -1) ||
                    // lastly check to see if the name was a levenshtein match
                    levenshtein(name, keys[i]) <= MAX_LEV_DISTANCE)) {
                    return false;
                }
            }
            return true;
        }

        function getQuery() {
            var matches, type, query, raw =
                document.getElementsByClassName('search-input')[0].value;
            query = raw;

            matches = query.match(/^(fn|mod|struct|enum|trait|type|const|macro)\s*:\s*/i);
            if (matches) {
                type = matches[1].replace(/^const$/, 'constant');
                query = query.substring(matches[0].length);
            }

            return {
                raw: raw,
                query: query,
                type: type,
                id: query + type
            };
        }

        function initSearchNav() {
            var hoverTimeout;

            var click_func = function(e) {
                var el = e.target;
                // to retrieve the real "owner" of the event.
                while (el.tagName !== 'TR') {
                    el = el.parentNode;
                }
                var dst = e.target.getElementsByTagName('a');
                if (dst.length < 1) {
                    return;
                }
                dst = dst[0];
                if (window.location.pathname === dst.pathname) {
                    addClass(document.getElementById('search'), 'hidden');
                    removeClass(document.getElementById('main'), 'hidden');
                    document.location.href = dst.href;
                }
            };
            var mouseover_func = function(e) {
                var el = e.target;
                // to retrieve the real "owner" of the event.
                while (el.tagName !== 'TR') {
                    el = el.parentNode;
                }
                clearTimeout(hoverTimeout);
                hoverTimeout = setTimeout(function() {
                    onEach(document.getElementsByClassName('search-results'), function(e) {
                        onEach(e.getElementsByClassName('result'), function(i_e) {
                            removeClass(i_e, 'highlighted');
                        });
                    });
                    addClass(el, 'highlighted');
                }, 20);
            };
            onEach(document.getElementsByClassName('search-results'), function(e) {
                onEach(e.getElementsByClassName('result'), function(i_e) {
                    i_e.onclick = click_func;
                    i_e.onmouseover = mouseover_func;
                });
            });

            var search_input = document.getElementsByClassName('search-input')[0];
            search_input.onkeydown = function(e) {
                // "actives" references the currently highlighted item in each search tab.
                // Each array in "actives" represents a tab.
                var actives = [[], [], []];
                // "current" is used to know which tab we're looking into.
                var current = 0;
                onEach(document.getElementsByClassName('search-results'), function(e) {
                    onEach(e.getElementsByClassName('highlighted'), function(e) {
                        actives[current].push(e);
                    });
                    current += 1;
                });

                if (e.which === 38) { // up
                    if (!actives[currentTab].length ||
                        !actives[currentTab][0].previousElementSibling) {
                        return;
                    }

                    addClass(actives[currentTab][0].previousElementSibling, 'highlighted');
                    removeClass(actives[currentTab][0], 'highlighted');
                } else if (e.which === 40) { // down
                    if (!actives[currentTab].length) {
                        var results = document.getElementsByClassName('search-results');
                        if (results.length > 0) {
                            var res = results[currentTab].getElementsByClassName('result');
                            if (res.length > 0) {
                                addClass(res[0], 'highlighted');
                            }
                        }
                    } else if (actives[currentTab][0].nextElementSibling) {
                        addClass(actives[currentTab][0].nextElementSibling, 'highlighted');
                        removeClass(actives[currentTab][0], 'highlighted');
                    }
                } else if (e.which === 13) { // return
                    if (actives[currentTab].length) {
                        document.location.href =
                            actives[currentTab][0].getElementsByTagName('a')[0].href;
                    }
                } else if (e.which === 9) { // tab
                    if (e.shiftKey) {
                        printTab(currentTab > 0 ? currentTab - 1 : 2);
                    } else {
                        printTab(currentTab > 1 ? 0 : currentTab + 1);
                    }
                    e.preventDefault();
                } else if (e.which === 16) { // shift
                    // Does nothing, it's just to avoid losing "focus" on the highlighted element.
                } else if (actives[currentTab].length > 0) {
                    removeClass(actives[currentTab][0], 'highlighted');
                }
            };
        }

        function escape(content) {
            var h1 = document.createElement('h1');
            h1.textContent = content;
            return h1.innerHTML;
        }

        function addTab(array, query, display) {
            var extraStyle = '';
            if (display === false) {
                extraStyle = ' style="display: none;"';
            }

            var output = '';
            if (array.length > 0) {
                output = '<table class="search-results"' + extraStyle + '>';
                var shown = [];

                array.forEach(function(item) {
                    var name, type, href, displayPath;

                    if (shown.indexOf(item) !== -1) {
                        return;
                    }

                    shown.push(item);
                    name = item.name;
                    type = itemTypes[item.ty];

                    if (type === 'mod') {
                        displayPath = item.path + '::';
                        href = rootPath + item.path.replace(/::/g, '/') + '/' +
                               name + '/index.html';
                    } else if (type === "primitive") {
                        displayPath = "";
                        href = rootPath + item.path.replace(/::/g, '/') +
                               '/' + type + '.' + name + '.html';
                    } else if (type === "externcrate") {
                        displayPath = "";
                        href = rootPath + name + '/index.html';
                    } else if (item.parent !== undefined) {
                        var myparent = item.parent;
                        var anchor = '#' + type + '.' + name;
                        var parentType = itemTypes[myparent.ty];
                        if (parentType === "primitive") {
                            displayPath = myparent.name + '::';
                        } else {
                            displayPath = item.path + '::' + myparent.name + '::';
                        }
                        href = rootPath + item.path.replace(/::/g, '/') +
                               '/' + parentType +
                               '.' + myparent.name +
                               '.html' + anchor;
                    } else {
                        displayPath = item.path + '::';
                        href = rootPath + item.path.replace(/::/g, '/') +
                               '/' + type + '.' + name + '.html';
                    }

                    output += '<tr class="' + type + ' result"><td>' +
                              '<a href="' + href + '">' +
                              displayPath + '<span class="' + type + '">' +
                              name + '</span></a></td><td>' +
                              '<a href="' + href + '">' +
                              '<span class="desc">' + escape(item.desc) +
                              '&nbsp;</span></a></td></tr>';
                });
                output += '</table>';
            } else {
                output = '<div class="search-failed"' + extraStyle + '>No results :(<br/>' +
                    'Try on <a href="https://duckduckgo.com/?q=' +
                    encodeURIComponent('rust ' + query.query) +
                    '">DuckDuckGo</a>?</div>';
            }
            return output;
        }

        function makeTabHeader(tabNb, text, nbElems) {
            if (currentTab === tabNb) {
                return '<div class="selected">' + text +
                       ' <div class="count">(' + nbElems + ')</div></div>';
            }
            return '<div>' + text + ' <div class="count">(' + nbElems + ')</div></div>';
        }

        function showResults(results) {
            var output, query = getQuery();

            currentResults = query.id;
            output = '<h1>Results for ' + escape(query.query) +
                (query.type ? ' (type: ' + escape(query.type) + ')' : '') + '</h1>' +
                '<div id="titles">' +
                makeTabHeader(0, "Types/modules", results['others'].length) +
                makeTabHeader(1, "As parameters", results['in_args'].length) +
                makeTabHeader(2, "As return value", results['returned'].length) +
                '</div><div id="results">';

            output += addTab(results['others'], query);
            output += addTab(results['in_args'], query, false);
            output += addTab(results['returned'], query, false);
            output += '</div>';

            addClass(document.getElementById('main'), 'hidden');
            var search = document.getElementById('search');
            removeClass(search, 'hidden');
            search.innerHTML = output;
            var tds = search.getElementsByTagName('td');
            var td_width = 0;
            if (tds.length > 0) {
                td_width = tds[0].offsetWidth;
            }
            var width = search.offsetWidth - 40 - td_width;
            onEach(search.getElementsByClassName('desc'), function(e) {
                e.style.width = width + 'px';
            });
            initSearchNav();
            var elems = document.getElementById('titles').childNodes;
            elems[0].onclick = function() { printTab(0); };
            elems[1].onclick = function() { printTab(1); };
            elems[2].onclick = function() { printTab(2); };
            printTab(currentTab);
        }

        function search(e) {
            var query,
                obj, i, len,
                results = {"in_args": [], "returned": [], "others": []},
                resultIndex;
            var params = getQueryStringParams();

            query = getQuery();
            if (e) {
                e.preventDefault();
            }

            if (!query.query || query.id === currentResults) {
                return;
            }

            // Update document title to maintain a meaningful browser history
            document.title = "Results for " + query.query + " - Rust";

            // Because searching is incremental by character, only the most
            // recent search query is added to the browser history.
            if (browserSupportsHistoryApi()) {
                if (!history.state && !params.search) {
                    history.pushState(query, "", "?search=" + encodeURIComponent(query.raw));
                } else {
                    history.replaceState(query, "", "?search=" + encodeURIComponent(query.raw));
                }
            }

            results = execQuery(query, 20000, index);
            showResults(results);
        }

        function itemTypeFromName(typename) {
            for (var i = 0; i < itemTypes.length; ++i) {
                if (itemTypes[i] === typename) {
                    return i;
                }
            }
            return -1;
        }

        function buildIndex(rawSearchIndex) {
            searchIndex = [];
            var searchWords = [];
            for (var crate in rawSearchIndex) {
                if (!rawSearchIndex.hasOwnProperty(crate)) { continue; }

                searchWords.push(crate);
                searchIndex.push({
                    crate: crate,
                    ty: 1, // == ExternCrate
                    name: crate,
                    path: "",
                    desc: rawSearchIndex[crate].doc,
                    type: null,
                });

                // an array of [(Number) item type,
                //              (String) name,
                //              (String) full path or empty string for previous path,
                //              (String) description,
                //              (Number | null) the parent path index to `paths`]
                //              (Object | null) the type of the function (if any)
                var items = rawSearchIndex[crate].items;
                // an array of [(Number) item type,
                //              (String) name]
                var paths = rawSearchIndex[crate].paths;

                // convert `paths` into an object form
                var len = paths.length;
                for (var i = 0; i < len; ++i) {
                    paths[i] = {ty: paths[i][0], name: paths[i][1]};
                }

                // convert `items` into an object form, and construct word indices.
                //
                // before any analysis is performed lets gather the search terms to
                // search against apart from the rest of the data.  This is a quick
                // operation that is cached for the life of the page state so that
                // all other search operations have access to this cached data for
                // faster analysis operations
                var len = items.length;
                var lastPath = "";
                for (var i = 0; i < len; ++i) {
                    var rawRow = items[i];
                    var row = {crate: crate, ty: rawRow[0], name: rawRow[1],
                               path: rawRow[2] || lastPath, desc: rawRow[3],
                               parent: paths[rawRow[4]], type: rawRow[5]};
                    searchIndex.push(row);
                    if (typeof row.name === "string") {
                        var word = row.name.toLowerCase();
                        searchWords.push(word);
                    } else {
                        searchWords.push("");
                    }
                    lastPath = row.path;
                }
            }
            return searchWords;
        }

        function startSearch() {
            var searchTimeout;
            var callback = function() {
                var search_input = document.getElementsByClassName('search-input');
                if (search_input.length < 1) { return; }
                search_input = search_input[0];
                clearTimeout(searchTimeout);
                if (search_input.value.length === 0) {
                    if (browserSupportsHistoryApi()) {
                        history.replaceState("", "std - Rust", "?search=");
                    }
                    var main = document.getElementById('main');
                    if (hasClass(main, 'content')) {
                        removeClass(main, 'hidden');
                    }
                    var search_c = document.getElementById('search');
                    if (hasClass(search_c, 'content')) {
                        addClass(search_c, 'hidden');
                    }
                } else {
                    searchTimeout = setTimeout(search, 500);
                }
            };
            var search_input = document.getElementsByClassName("search-input")[0];
            search_input.onkeyup = callback;
            search_input.oninput = callback;
            document.getElementsByClassName("search-form")[0].onsubmit = function(e) {
                e.preventDefault();
                clearTimeout(searchTimeout);
                search();
            };
            search_input.onchange = function(e) {
                // Do NOT e.preventDefault() here. It will prevent pasting.
                clearTimeout(searchTimeout);
                // zero-timeout necessary here because at the time of event handler execution the
                // pasted content is not in the input field yet. Shouldn’t make any difference for
                // change, though.
                setTimeout(search, 0);
            };
            search_input.onpaste = search_input.onchange;

            // Push and pop states are used to add search results to the browser
            // history.
            if (browserSupportsHistoryApi()) {
                // Store the previous <title> so we can revert back to it later.
                var previousTitle = document.title;

                window.onpopstate = function(e) {
                    var params = getQueryStringParams();
                    // When browsing back from search results the main page
                    // visibility must be reset.
                    if (!params.search) {
                        var main = document.getElementById('main');
                        if (hasClass(main, 'content')) {
                            removeClass(main, 'hidden');
                        }
                        var search_c = document.getElementById('search');
                        if (hasClass(search_c, 'content')) {
                            addClass(search_c, 'hidden');
                        }
                    }
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
                    if (params.search) {
                        document.getElementsByClassName('search-input')[0].value = params.search;
                    } else {
                        document.getElementsByClassName('search-input')[0].value = '';
                    }
                    // Some browsers fire 'onpopstate' for every page load
                    // (Chrome), while others fire the event only when actually
                    // popping a state (Firefox), which is why search() is
                    // called both here and at the end of the startSearch()
                    // function.
                    search();
                };
            }
            search();
        }

        index = buildIndex(rawSearchIndex);
        startSearch();

        // Draw a convenient sidebar of known crates if we have a listing
        if (rootPath === '../') {
            var sidebar = document.getElementsByClassName('sidebar')[0];
            var div = document.createElement('div');
            div.className = 'block crate';
            div.innerHTML = '<h3>Crates</h3>';
            var ul = document.createElement('ul');
            div.appendChild(ul);

            var crates = [];
            for (var crate in rawSearchIndex) {
                if (!rawSearchIndex.hasOwnProperty(crate)) {
                    continue;
                }
                crates.push(crate);
            }
            crates.sort();
            for (var i = 0; i < crates.length; ++i) {
                var klass = 'crate';
                if (crates[i] === window.currentCrate) {
                    klass += ' current';
                }
                var link = document.createElement('a');
                link.href = '../' + crates[i] + '/index.html';
                link.title = rawSearchIndex[crates[i]].doc;
                link.className = klass;
                link.textContent = crates[i];

                var li = document.createElement('li');
                li.appendChild(link);
                ul.appendChild(li);
            }
            sidebar.appendChild(div);
        }
    }

    window.initSearch = initSearch;

    // delayed sidebar rendering.
    function initSidebarItems(items) {
        var sidebar = document.getElementsByClassName('sidebar')[0];
        var current = window.sidebarCurrent;

        function block(shortty, longty) {
            var filtered = items[shortty];
            if (!filtered) { return; }

            var div = document.createElement('div');
            div.className = 'block ' + shortty;
            var h3 = document.createElement('h3');
            h3.textContent = longty;
            div.appendChild(h3);
            var ul = document.createElement('ul');

            for (var i = 0; i < filtered.length; ++i) {
                var item = filtered[i];
                var name = item[0];
                var desc = item[1]; // can be null

                var klass = shortty;
                if (name === current.name && shortty === current.ty) {
                    klass += ' current';
                }
                var path;
                if (shortty === 'mod') {
                    path = name + '/index.html';
                } else {
                    path = shortty + '.' + name + '.html';
                }
                var link = document.createElement('a');
                link.href = current.relpath + path;
                link.title = desc;
                link.className = klass;
                link.textContent = name;
                var li = document.createElement('li');
                li.appendChild(link);
                ul.appendChild(li);
            }
            div.appendChild(ul);
            sidebar.appendChild(div);
        }

        block("primitive", "Primitive Types");
        block("mod", "Modules");
        block("macro", "Macros");
        block("struct", "Structs");
        block("enum", "Enums");
        block("union", "Unions");
        block("constant", "Constants");
        block("static", "Statics");
        block("trait", "Traits");
        block("fn", "Functions");
        block("type", "Type Definitions");
        block("foreigntype", "Foreign Types");
    }

    window.initSidebarItems = initSidebarItems;

    window.register_implementors = function(imp) {
        var list = document.getElementById('implementors-list');
        var libs = Object.getOwnPropertyNames(imp);
        for (var i = 0; i < libs.length; ++i) {
            if (libs[i] === currentCrate) { continue; }
            var structs = imp[libs[i]];
            for (var j = 0; j < structs.length; ++j) {
                var code = document.createElement('code');
                code.innerHTML = structs[j];

                var x = code.getElementsByTagName('a');
                for (var k = 0; k < x.length; k++) {
                    var href = x[k].getAttribute('href');
                    if (href && href.indexOf('http') !== 0) {
                        x[k].setAttribute('href', rootPath + href);
                    }
                }
                var li = document.createElement('li');
                li.appendChild(code);
                list.appendChild(li);
            }
        }
    };
    if (window.pending_implementors) {
        window.register_implementors(window.pending_implementors);
    }

    function labelForToggleButton(sectionIsCollapsed) {
        if (sectionIsCollapsed) {
            // button will expand the section
            return "+";
        }
        // button will collapse the section
        // note that this text is also set in the HTML template in render.rs
        return "\u2212"; // "\u2212" is '−' minus sign
    }

    function onEveryMatchingChild(elem, className, func) {
        if (elem && className && func) {
            for (var i = 0; i < elem.childNodes.length; i++) {
                if (hasClass(elem.childNodes[i], className)) {
                    func(elem.childNodes[i]);
                } else {
                    onEveryMatchingChild(elem.childNodes[i], className, func);
                }
            }
        }
    }

    function toggleAllDocs() {
        var toggle = document.getElementById("toggle-all-docs");
        if (hasClass(toggle, "will-expand")) {
            removeClass(toggle, "will-expand");
            onEveryMatchingChild(toggle, "inner", function(e) {
                e.innerHTML = labelForToggleButton(false);
            });
            toggle.title = "collapse all docs";
            onEach(document.getElementsByClassName("docblock"), function(e) {
                e.style.display = 'block';
            });
            onEach(document.getElementsByClassName("toggle-label"), function(e) {
                e.style.display = 'none';
            });
            onEach(document.getElementsByClassName("toggle-wrapper"), function(e) {
                removeClass(e, "collapsed");
            });
            onEach(document.getElementsByClassName("collapse-toggle"), function(e) {
                onEveryMatchingChild(e, "inner", function(i_e) {
                    i_e.innerHTML = labelForToggleButton(false);
                });
            });
        } else {
            addClass(toggle, "will-expand");
            onEveryMatchingChild(toggle, "inner", function(e) {
                e.innerHTML = labelForToggleButton(true);
            });
            toggle.title = "expand all docs";
            onEach(document.getElementsByClassName("docblock"), function(e) {
                e.style.display = 'none';
            });
            onEach(document.getElementsByClassName("toggle-label"), function(e) {
                e.style.display = 'inline-block';
            });
            onEach(document.getElementsByClassName("toggle-wrapper"), function(e) {
                addClass(e, "collapsed");
            });
            onEach(document.getElementsByClassName("collapse-toggle"), function(e) {
                onEveryMatchingChild(e, "inner", function(i_e) {
                    i_e.innerHTML = labelForToggleButton(true);
                });
            });
        }
    }

    function collapseDocs(toggle) {
        if (!toggle || !toggle.parentNode) {
            return;
        }
        var relatedDoc = toggle.parentNode.nextElementSibling;
        if (hasClass(relatedDoc, "stability")) {
            relatedDoc = relatedDoc.nextElementSibling;
        }
        if (hasClass(relatedDoc, "docblock")) {
            if (!isHidden(relatedDoc)) {
                relatedDoc.style.display = 'none';
                onEach(toggle.childNodes, function(e) {
                    if (hasClass(e, 'toggle-label')) {
                        e.style.display = 'inline-block';
                    }
                    if (hasClass(e, 'inner')) {
                        e.innerHTML = labelForToggleButton(true);
                    }
                });
                addClass(toggle.parentNode, 'collapsed');
            } else {
                relatedDoc.style.display = 'block';
                removeClass(toggle.parentNode, 'collapsed');
                onEach(toggle.childNodes, function(e) {
                    if (hasClass(e, 'toggle-label')) {
                        e.style.display = 'none';
                    }
                    if (hasClass(e, 'inner')) {
                        e.innerHTML = labelForToggleButton(false);
                    }
                });
            }
        }
    }

    var x = document.getElementById('toggle-all-docs');
    if (x) {
        x.onclick = toggleAllDocs;
    }

    function insertAfter(newNode, referenceNode) {
        referenceNode.parentNode.insertBefore(newNode, referenceNode.nextSibling);
    }

    var toggle = document.createElement('a');
    toggle.href = 'javascript:void(0)';
    toggle.className = 'collapse-toggle';
    toggle.innerHTML = "[<span class='inner'>"+labelForToggleButton(false)+"</span>]";

    var func = function(e) {
        var next = e.nextElementSibling;
        if (!next) {
            return;
        }
        if (hasClass(next, 'docblock') ||
            (hasClass(next, 'stability') &&
             hasClass(next.nextElementSibling, 'docblock'))) {
            insertAfter(toggle.cloneNode(true), e.childNodes[e.childNodes.length - 1]);
        }
    }
    onEach(document.getElementsByClassName('method'), func);
    onEach(document.getElementsByClassName('impl-items'), function(e) {
        onEach(e.getElementsByClassName('associatedconstant'), func);
    });

    function createToggle() {
        var span = document.createElement('span');
        span.className = 'toggle-label';
        span.style.display = 'none';
        span.innerHTML = '&nbsp;Expand&nbsp;description';

        var mainToggle = toggle.cloneNode(true);
        mainToggle.appendChild(span);

        var wrapper = document.createElement('div');
        wrapper.className = 'toggle-wrapper';
        wrapper.appendChild(mainToggle);
        return wrapper;
    }

    onEach(document.getElementById('main').getElementsByClassName('docblock'), function(e) {
        if (e.parentNode.id === "main") {
            e.parentNode.insertBefore(createToggle(), e);
        }
    });

    onEach(document.getElementsByClassName('docblock'), function(e) {
        if (hasClass(e, 'autohide')) {
            var wrap = e.previousElementSibling;
            if (wrap && hasClass(wrap, 'toggle-wrapper')) {
                var toggle = wrap.childNodes[0];
                if (e.childNodes[0].tagName === 'H3') {
                    onEach(toggle.getElementsByClassName('toggle-label'), function(i_e) {
                        i_e.innerHTML = " Show " + e.childNodes[0].innerHTML;
                    });
                }
                e.style.display = 'none';
                addClass(wrap, 'collapsed');
                onEach(toggle.getElementsByClassName('inner'), function(e) {
                    e.innerHTML = labelForToggleButton(true);
                });
                onEach(toggle.getElementsByClassName('toggle-label'), function(e) {
                    e.style.display = 'inline-block';
                });
            }
        }
    })

    function createToggleWrapper() {
        var span = document.createElement('span');
        span.className = 'toggle-label';
        span.style.display = 'none';
        span.innerHTML = '&nbsp;Expand&nbsp;attributes';
        toggle.appendChild(span);

        var wrapper = document.createElement('div');
        wrapper.className = 'toggle-wrapper toggle-attributes';
        wrapper.appendChild(toggle);
        return wrapper;
    }

    // In the search display, allows to switch between tabs.
    function printTab(nb) {
        if (nb === 0 || nb === 1 || nb === 2) {
            currentTab = nb;
        }
        var nb_copy = nb;
        onEach(document.getElementById('titles').childNodes, function(elem) {
            if (nb_copy === 0) {
                addClass(elem, 'selected');
            } else {
                removeClass(elem, 'selected');
            }
            nb_copy -= 1;
        });
        onEach(document.getElementById('results').childNodes, function(elem) {
            if (nb === 0) {
                elem.style.display = '';
            } else {
                elem.style.display = 'none';
            }
            nb -= 1;
        });
    }

    onEach(document.getElementById('main').getElementsByTagName('pre'), function(e) {
        onEach(e.getElementsByClassName('attributes'), function(i_e) {
            i_e.parentNode.insertBefore(createToggleWrapper(), i_e);
            collapseDocs(i_e.previousSibling.childNodes[0]);
        });
    });

    onEach(document.getElementsByClassName('rust-example-rendered'), function(e) {
        if (hasClass(e, 'compile_fail')) {
            e.addEventListener("mouseover", function(event) {
                e.previousElementSibling.childNodes[0].style.color = '#f00';
            });
            e.addEventListener("mouseout", function(event) {
                e.previousElementSibling.childNodes[0].style.color = '';
            });
        } else if (hasClass(e, 'ignore')) {
            e.addEventListener("mouseover", function(event) {
                e.previousElementSibling.childNodes[0].style.color = '#ff9200';
            });
            e.addEventListener("mouseout", function(event) {
                e.previousElementSibling.childNodes[0].style.color = '';
            });
        }
    });

    var search_input = document.getElementsByClassName("search-input")[0];

    if (search_input) {
        search_input.onfocus = function() {
            if (search_input.value !== "") {
                addClass(document.getElementById("main"), "hidden");
                removeClass(document.getElementById("search"), "hidden");
                if (browserSupportsHistoryApi()) {
                    history.replaceState(search_input.value,
                                         "",
                                         "?search=" + encodeURIComponent(search_input.value));
                }
            }
        };
    }
}());

// Sets the focus on the search bar at the top of the page
function focusSearchBar() {
    document.getElementsByClassName('search-input')[0].focus();
}
