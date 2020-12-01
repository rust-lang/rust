// From rust:
/* global ALIASES, currentCrate, rootPath */

// Local js definitions:
/* global addClass, getCurrentValue, hasClass */
/* global onEachLazy, hasOwnProperty, removeClass, updateLocalStorage */
/* global hideThemeButtonState, showThemeButtonState */

if (!String.prototype.startsWith) {
    String.prototype.startsWith = function(searchString, position) {
        position = position || 0;
        return this.indexOf(searchString, position) === position;
    };
}
if (!String.prototype.endsWith) {
    String.prototype.endsWith = function(suffix, length) {
        var l = length || this.length;
        return this.indexOf(suffix, l - suffix.length) !== -1;
    };
}

if (!DOMTokenList.prototype.add) {
    DOMTokenList.prototype.add = function(className) {
        if (className && !hasClass(this, className)) {
            if (this.className && this.className.length > 0) {
                this.className += " " + className;
            } else {
                this.className = className;
            }
        }
    };
}

if (!DOMTokenList.prototype.remove) {
    DOMTokenList.prototype.remove = function(className) {
        if (className && this.className) {
            this.className = (" " + this.className + " ").replace(" " + className + " ", " ")
                                                         .trim();
        }
    };
}

function getSearchInput() {
    return document.getElementsByClassName("search-input")[0];
}

function getSearchElement() {
    return document.getElementById("search");
}

function getThemesElement() {
    return document.getElementById("theme-choices");
}

function getThemePickerElement() {
    return document.getElementById("theme-picker");
}

// Sets the focus on the search bar at the top of the page
function focusSearchBar() {
    getSearchInput().focus();
}

// Removes the focus from the search bar
function defocusSearchBar() {
    getSearchInput().blur();
}

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
                     "foreigntype",
                     "keyword",
                     "existential",
                     "attr",
                     "derive",
                     "traitalias"];

    var disableShortcuts = getSettingValue("disable-shortcuts") === "true";
    var search_input = getSearchInput();
    var searchTimeout = null;
    var toggleAllDocsId = "toggle-all-docs";

    // On the search screen, so you remain on the last tab you opened.
    //
    // 0 for "In Names"
    // 1 for "In Parameters"
    // 2 for "In Return Types"
    var currentTab = 0;

    var mouseMovedAfterSearch = true;

    var titleBeforeSearch = document.title;
    var searchTitle = null;

    function clearInputTimeout() {
        if (searchTimeout !== null) {
            clearTimeout(searchTimeout);
            searchTimeout = null;
        }
    }

    function getPageId() {
        if (window.location.hash) {
            var tmp = window.location.hash.replace(/^#/, "");
            if (tmp.length > 0) {
                return tmp;
            }
        }
        return null;
    }

    function showSidebar() {
        var elems = document.getElementsByClassName("sidebar-elems")[0];
        if (elems) {
            addClass(elems, "show-it");
        }
        var sidebar = document.getElementsByClassName("sidebar")[0];
        if (sidebar) {
            addClass(sidebar, "mobile");
            var filler = document.getElementById("sidebar-filler");
            if (!filler) {
                var div = document.createElement("div");
                div.id = "sidebar-filler";
                sidebar.appendChild(div);
            }
        }
    }

    function hideSidebar() {
        var elems = document.getElementsByClassName("sidebar-elems")[0];
        if (elems) {
            removeClass(elems, "show-it");
        }
        var sidebar = document.getElementsByClassName("sidebar")[0];
        removeClass(sidebar, "mobile");
        var filler = document.getElementById("sidebar-filler");
        if (filler) {
            filler.remove();
        }
        document.getElementsByTagName("body")[0].style.marginTop = "";
    }

    function showSearchResults(search) {
        if (search === null || typeof search === 'undefined') {
            search = getSearchElement();
        }
        addClass(main, "hidden");
        removeClass(search, "hidden");
        mouseMovedAfterSearch = false;
        document.title = searchTitle;
    }

    function hideSearchResults(search) {
        if (search === null || typeof search === 'undefined') {
            search = getSearchElement();
        }
        addClass(search, "hidden");
        removeClass(main, "hidden");
        document.title = titleBeforeSearch;
    }

    // used for special search precedence
    var TY_PRIMITIVE = itemTypes.indexOf("primitive");
    var TY_KEYWORD = itemTypes.indexOf("keyword");

    function getQueryStringParams() {
        var params = {};
        window.location.search.substring(1).split("&").
            map(function(s) {
                var pair = s.split("=");
                params[decodeURIComponent(pair[0])] =
                    typeof pair[1] === "undefined" ? null : decodeURIComponent(pair[1]);
            });
        return params;
    }

    function browserSupportsHistoryApi() {
        return window.history && typeof window.history.pushState === "function";
    }

    function isHidden(elem) {
        return elem.offsetHeight === 0;
    }

    var main = document.getElementById("main");
    var savedHash = "";

    function handleHashes(ev) {
        var elem;
        var search = getSearchElement();
        if (ev !== null && search && !hasClass(search, "hidden") && ev.newURL) {
            // This block occurs when clicking on an element in the navbar while
            // in a search.
            hideSearchResults(search);
            var hash = ev.newURL.slice(ev.newURL.indexOf("#") + 1);
            if (browserSupportsHistoryApi()) {
                history.replaceState(hash, "", "?search=#" + hash);
            }
            elem = document.getElementById(hash);
            if (elem) {
                elem.scrollIntoView();
            }
        }
        // This part is used in case an element is not visible.
        if (savedHash !== window.location.hash) {
            savedHash = window.location.hash;
            if (savedHash.length === 0) {
                return;
            }
            elem = document.getElementById(savedHash.slice(1)); // we remove the '#'
            if (!elem || !isHidden(elem)) {
                return;
            }
            var parent = elem.parentNode;
            if (parent && hasClass(parent, "impl-items")) {
                // In case this is a trait implementation item, we first need to toggle
                // the "Show hidden undocumented items".
                onEachLazy(parent.getElementsByClassName("collapsed"), function(e) {
                    if (e.parentNode === parent) {
                        // Only click on the toggle we're looking for.
                        e.click();
                        return true;
                    }
                });
                if (isHidden(elem)) {
                    // The whole parent is collapsed. We need to click on its toggle as well!
                    if (hasClass(parent.lastElementChild, "collapse-toggle")) {
                        parent.lastElementChild.click();
                    }
                }
            }
        }
    }

    function highlightSourceLines(match, ev) {
        if (typeof match === "undefined") {
            // If we're in mobile mode, we should hide the sidebar in any case.
            hideSidebar();
            match = window.location.hash.match(/^#?(\d+)(?:-(\d+))?$/);
        }
        if (!match) {
            return;
        }
        var from = parseInt(match[1], 10);
        var to = from;
        if (typeof match[2] !== "undefined") {
            to = parseInt(match[2], 10);
        }
        if (to < from) {
            var tmp = to;
            to = from;
            from = tmp;
        }
        var elem = document.getElementById(from);
        if (!elem) {
            return;
        }
        if (!ev) {
            var x = document.getElementById(from);
            if (x) {
                x.scrollIntoView();
            }
        }
        onEachLazy(document.getElementsByClassName("line-numbers"), function(e) {
            onEachLazy(e.getElementsByTagName("span"), function(i_e) {
                removeClass(i_e, "line-highlighted");
            });
        });
        for (var i = from; i <= to; ++i) {
            elem = document.getElementById(i);
            if (!elem) {
                break;
            }
            addClass(elem, "line-highlighted");
        }
    }

    function onHashChange(ev) {
        // If we're in mobile mode, we should hide the sidebar in any case.
        hideSidebar();
        var match = window.location.hash.match(/^#?(\d+)(?:-(\d+))?$/);
        if (match) {
            return highlightSourceLines(match, ev);
        }
        handleHashes(ev);
    }

    function expandSection(id) {
        var elem = document.getElementById(id);
        if (elem && isHidden(elem)) {
            var h3 = elem.parentNode.previousElementSibling;
            if (h3 && h3.tagName !== "H3") {
                h3 = h3.previousElementSibling; // skip div.docblock
            }

            if (h3) {
                var collapses = h3.getElementsByClassName("collapse-toggle");
                if (collapses.length > 0) {
                    // The element is not visible, we need to make it appear!
                    collapseDocs(collapses[0], "show");
                }
            }
        }
    }

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
        if ("key" in ev && typeof ev.key != "undefined") {
            return ev.key;
        }

        var c = ev.charCode || ev.keyCode;
        if (c == 27) {
            return "Escape";
        }
        return String.fromCharCode(c);
    }

    function getHelpElement() {
        buildHelperPopup();
        return document.getElementById("help");
    }

    function displayHelp(display, ev, help) {
        help = help ? help : getHelpElement();
        if (display === true) {
            if (hasClass(help, "hidden")) {
                ev.preventDefault();
                removeClass(help, "hidden");
                addClass(document.body, "blur");
            }
        } else if (hasClass(help, "hidden") === false) {
            ev.preventDefault();
            addClass(help, "hidden");
            removeClass(document.body, "blur");
        }
    }

    function handleEscape(ev) {
        var help = getHelpElement();
        var search = getSearchElement();
        if (hasClass(help, "hidden") === false) {
            displayHelp(false, ev, help);
        } else if (hasClass(search, "hidden") === false) {
            clearInputTimeout();
            ev.preventDefault();
            hideSearchResults(search);
        }
        defocusSearchBar();
        hideThemeButtonState();
    }

    function handleShortcut(ev) {
        // Don't interfere with browser shortcuts
        if (ev.ctrlKey || ev.altKey || ev.metaKey || disableShortcuts === true) {
            return;
        }

        if (document.activeElement.tagName === "INPUT") {
            switch (getVirtualKey(ev)) {
            case "Escape":
                handleEscape(ev);
                break;
            }
        } else {
            switch (getVirtualKey(ev)) {
            case "Escape":
                handleEscape(ev);
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
                displayHelp(true, ev);
                break;

            case "t":
            case "T":
                displayHelp(false, ev);
                ev.preventDefault();
                var themePicker = getThemePickerElement();
                themePicker.click();
                themePicker.focus();
                break;

            default:
                var themePicker = getThemePickerElement();
                if (themePicker.parentNode.contains(ev.target)) {
                    handleThemeKeyDown(ev);
                }
            }
        }
    }

    function handleThemeKeyDown(ev) {
        var active = document.activeElement;
        var themes = getThemesElement();
        switch (getVirtualKey(ev)) {
        case "ArrowUp":
            ev.preventDefault();
            if (active.previousElementSibling && ev.target.id !== "theme-picker") {
                active.previousElementSibling.focus();
            } else {
                showThemeButtonState();
                themes.lastElementChild.focus();
            }
            break;
        case "ArrowDown":
            ev.preventDefault();
            if (active.nextElementSibling && ev.target.id !== "theme-picker") {
                active.nextElementSibling.focus();
            } else {
                showThemeButtonState();
                themes.firstElementChild.focus();
            }
            break;
        case "Enter":
        case "Return":
        case "Space":
            if (ev.target.id === "theme-picker" && themes.style.display === "none") {
                ev.preventDefault();
                showThemeButtonState();
                themes.firstElementChild.focus();
            }
            break;
        case "Home":
            ev.preventDefault();
            themes.firstElementChild.focus();
            break;
        case "End":
            ev.preventDefault();
            themes.lastElementChild.focus();
            break;
        // The escape key is handled in handleEscape, not here,
        // so that pressing escape will close the menu even if it isn't focused
        }
    }

    function findParentElement(elem, tagName) {
        do {
            if (elem && elem.tagName === tagName) {
                return elem;
            }
            elem = elem.parentNode;
        } while (elem);
        return null;
    }

    document.addEventListener("keypress", handleShortcut);
    document.addEventListener("keydown", handleShortcut);

    function resetMouseMoved(ev) {
        mouseMovedAfterSearch = true;
    }

    document.addEventListener("mousemove", resetMouseMoved);

    var handleSourceHighlight = (function() {
        var prev_line_id = 0;

        var set_fragment = function(name) {
            var x = window.scrollX,
                y = window.scrollY;
            if (browserSupportsHistoryApi()) {
                history.replaceState(null, null, "#" + name);
                highlightSourceLines();
            } else {
                location.replace("#" + name);
            }
            // Prevent jumps when selecting one or many lines
            window.scrollTo(x, y);
        };

        return function(ev) {
            var cur_line_id = parseInt(ev.target.id, 10);
            ev.preventDefault();

            if (ev.shiftKey && prev_line_id) {
                // Swap selection if needed
                if (prev_line_id > cur_line_id) {
                    var tmp = prev_line_id;
                    prev_line_id = cur_line_id;
                    cur_line_id = tmp;
                }

                set_fragment(prev_line_id + "-" + cur_line_id);
            } else {
                prev_line_id = cur_line_id;

                set_fragment(cur_line_id);
            }
        };
    }());

    document.addEventListener("click", function(ev) {
        if (hasClass(ev.target, "help-button")) {
            displayHelp(true, ev);
        } else if (hasClass(ev.target, "collapse-toggle")) {
            collapseDocs(ev.target, "toggle");
        } else if (hasClass(ev.target.parentNode, "collapse-toggle")) {
            collapseDocs(ev.target.parentNode, "toggle");
        } else if (ev.target.tagName === "SPAN" && hasClass(ev.target.parentNode, "line-numbers")) {
            handleSourceHighlight(ev);
        } else if (hasClass(getHelpElement(), "hidden") === false) {
            var help = getHelpElement();
            var is_inside_help_popup = ev.target !== help && help.contains(ev.target);
            if (is_inside_help_popup === false) {
                addClass(help, "hidden");
                removeClass(document.body, "blur");
            }
        } else {
            // Making a collapsed element visible on onhashchange seems
            // too late
            var a = findParentElement(ev.target, "A");
            if (a && a.hash) {
                expandSection(a.hash.replace(/^#/, ""));
            }
        }
    });

    (function() {
        var x = document.getElementsByClassName("version-selector");
        if (x.length > 0) {
            x[0].onchange = function() {
                var i, match,
                    url = document.location.href,
                    stripped = "",
                    len = rootPath.match(/\.\.\//g).length + 1;

                for (i = 0; i < len; ++i) {
                    match = url.match(/\/[^\/]*$/);
                    if (i < len - 1) {
                        stripped = match[0] + stripped;
                    }
                    url = url.substring(0, url.length - match[0].length);
                }

                var selectedVersion = document.getElementsByClassName("version-selector")[0].value;
                url += "/" + selectedVersion + stripped;

                document.location.href = url;
            };
        }
    }());

    /**
     * A function to compute the Levenshtein distance between two strings
     * Licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported
     * Full License can be found at http://creativecommons.org/licenses/by-sa/3.0/legalcode
     * This code is an unmodified version of the code written by Marco de Wit
     * and was found at http://stackoverflow.com/a/18514751/745719
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
        var GENERICS_DATA = 1;
        var NAME = 0;
        var INPUTS_DATA = 0;
        var OUTPUT_DATA = 1;
        var NO_TYPE_FILTER = -1;
        var currentResults, index, searchIndex;
        var ALIASES = {};
        var params = getQueryStringParams();

        // Populate search bar with query string search term when provided,
        // but only if the input bar is empty. This avoid the obnoxious issue
        // where you start trying to do a search, and the index loads, and
        // suddenly your search is gone!
        if (search_input.value === "") {
            search_input.value = params.search || "";
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
                var length = itemTypes.length;
                for (var i = 0; i < length; ++i) {
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

            var length = split.length;
            for (var z = 0; z < length; ++z) {
                if (split[z] === "") {
                    split.splice(z, 1);
                    z -= 1;
                }
            }

            function transformResults(results, isType) {
                var out = [];
                var length = results.length;
                for (var i = 0; i < length; ++i) {
                    if (results[i].id > -1) {
                        var obj = searchIndex[results[i].id];
                        obj.lev = results[i].lev;
                        if (isType !== true || obj.type) {
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
                }
                return out;
            }

            function sortResults(results, isType) {
                var ar = [];
                for (var entry in results) {
                    if (hasOwnProperty(results, entry)) {
                        ar.push(results[entry]);
                    }
                }
                results = ar;
                var i;
                var nresults = results.length;
                for (i = 0; i < nresults; ++i) {
                    results[i].word = searchWords[results[i].id];
                    results[i].item = searchIndex[results[i].id] || {};
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

                var length = results.length;
                for (i = 0; i < length; ++i) {
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

            function getObjectFromId(id) {
                if (typeof id === "number") {
                    return searchIndex[id];
                }
                return {'name': id};
            }

            function checkGenerics(obj, val) {
                // The names match, but we need to be sure that all generics kinda
                // match as well.
                var lev_distance = MAX_LEV_DISTANCE + 1;
                if (val.generics.length > 0) {
                    if (obj.length > GENERICS_DATA &&
                          obj[GENERICS_DATA].length >= val.generics.length) {
                        var elems = obj[GENERICS_DATA].slice(0);
                        var total = 0;
                        var done = 0;
                        // We need to find the type that matches the most to remove it in order
                        // to move forward.
                        var vlength = val.generics.length;
                        for (var y = 0; y < vlength; ++y) {
                            var lev = { pos: -1, lev: MAX_LEV_DISTANCE + 1};
                            var elength = elems.length;
                            var firstGeneric = getObjectFromId(val.generics[y]).name;
                            for (var x = 0; x < elength; ++x) {
                                var tmp_lev = levenshtein(getObjectFromId(elems[x]).name,
                                                          firstGeneric);
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
                        return Math.ceil(total / done);
                    }
                }
                return MAX_LEV_DISTANCE + 1;
            }

            // Check for type name and type generics (if any).
            function checkType(obj, val, literalSearch) {
                var lev_distance = MAX_LEV_DISTANCE + 1;
                var x;
                if (obj[NAME] === val.name) {
                    if (literalSearch === true) {
                        if (val.generics && val.generics.length !== 0) {
                            if (obj.length > GENERICS_DATA &&
                                  obj[GENERICS_DATA].length >= val.generics.length) {
                                var elems = obj[GENERICS_DATA].slice(0);
                                var allFound = true;

                                for (var y = 0; allFound === true && y < val.generics.length; ++y) {
                                    allFound = false;
                                    var firstGeneric = getObjectFromId(val.generics[y]).name;
                                    for (x = 0; allFound === false && x < elems.length; ++x) {
                                        allFound = getObjectFromId(elems[x]).name === firstGeneric;
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
                    if (obj.length > GENERICS_DATA && obj[GENERICS_DATA].length !== 0) {
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
                     if (obj.length > GENERICS_DATA && obj[GENERICS_DATA].length > 0) {
                        var length = obj[GENERICS_DATA].length;
                        for (x = 0; x < length; ++x) {
                            if (obj[GENERICS_DATA][x] === val.name) {
                                return true;
                            }
                        }
                    }
                    return false;
                }
                lev_distance = Math.min(levenshtein(obj[NAME], val.name), lev_distance);
                if (lev_distance <= MAX_LEV_DISTANCE) {
                    // The generics didn't match but the name kinda did so we give it
                    // a levenshtein distance value that isn't *this* good so it goes
                    // into the search results but not too high.
                    lev_distance = Math.ceil((checkGenerics(obj, val) + lev_distance) / 2);
                } else if (obj.length > GENERICS_DATA && obj[GENERICS_DATA].length > 0) {
                    // We can check if the type we're looking for is inside the generics!
                    var olength = obj[GENERICS_DATA].length;
                    for (x = 0; x < olength; ++x) {
                        lev_distance = Math.min(levenshtein(obj[GENERICS_DATA][x], val.name),
                                                lev_distance);
                    }
                }
                // Now whatever happens, the returned distance is "less good" so we should mark it
                // as such, and so we add 1 to the distance to make it "less good".
                return lev_distance + 1;
            }

            function findArg(obj, val, literalSearch, typeFilter) {
                var lev_distance = MAX_LEV_DISTANCE + 1;

                if (obj && obj.type && obj.type[INPUTS_DATA] && obj.type[INPUTS_DATA].length > 0) {
                    var length = obj.type[INPUTS_DATA].length;
                    for (var i = 0; i < length; i++) {
                        var tmp = obj.type[INPUTS_DATA][i];
                        if (typePassesFilter(typeFilter, tmp[1]) === false) {
                            continue;
                        }
                        tmp = checkType(tmp, val, literalSearch);
                        if (literalSearch === true) {
                            if (tmp === true) {
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
                return literalSearch === true ? false : lev_distance;
            }

            function checkReturned(obj, val, literalSearch, typeFilter) {
                var lev_distance = MAX_LEV_DISTANCE + 1;

                if (obj && obj.type && obj.type.length > OUTPUT_DATA) {
                    var ret = obj.type[OUTPUT_DATA];
                    if (typeof ret[0] === "string") {
                        ret = [ret];
                    }
                    for (var x = 0; x < ret.length; ++x) {
                        var tmp = ret[x];
                        if (typePassesFilter(typeFilter, tmp[1]) === false) {
                            continue;
                        }
                        tmp = checkType(tmp, val, literalSearch);
                        if (literalSearch === true) {
                            if (tmp === true) {
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
                return literalSearch === true ? false : lev_distance;
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
                    if (aborted === false) {
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

            function generateId(ty) {
                if (ty.parent && ty.parent.name) {
                    return itemTypes[ty.ty] + ty.path + ty.parent.name + ty.name;
                }
                return itemTypes[ty.ty] + ty.path + ty.name;
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
                var i;
                if (filterCrates !== undefined) {
                    if (ALIASES[filterCrates] && ALIASES[filterCrates][query.search]) {
                        for (i = 0; i < ALIASES[filterCrates][query.search].length; ++i) {
                            aliases.push(
                                createAliasFromItem(
                                    searchIndex[ALIASES[filterCrates][query.search][i]]));
                        }
                    }
                } else {
                    Object.keys(ALIASES).forEach(function(crate) {
                        if (ALIASES[crate][query.search]) {
                            var pushTo = crate === window.currentCrate ? crateAliases : aliases;
                            for (i = 0; i < ALIASES[crate][query.search].length; ++i) {
                                pushTo.push(
                                    createAliasFromItem(
                                        searchIndex[ALIASES[crate][query.search][i]]));
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
            var i;
            var ty;
            var fullId;
            var returned;
            var in_args;
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
                    fullId = generateId(ty);

                    if (searchWords[i] === val.name
                        && typePassesFilter(typeFilter, searchIndex[i].ty)
                        && results[fullId] === undefined) {
                        results[fullId] = {
                            id: i,
                            index: -1,
                            dontValidate: true,
                        };
                    }
                    if (in_args === true && results_in_args[fullId] === undefined) {
                        results_in_args[fullId] = {
                            id: i,
                            index: -1,
                            dontValidate: true,
                        };
                    }
                    if (returned === true && results_returned[fullId] === undefined) {
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
                for (i = 0; i < inputs.length; ++i) {
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
                    fullId = generateId(ty);

                    returned = checkReturned(ty, output, true, NO_TYPE_FILTER);
                    if (output.name === "*" || returned === true) {
                        in_args = false;
                        var is_module = false;

                        if (input === "*") {
                            is_module = true;
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
                        if (is_module === true) {
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
                var contains = paths.slice(0, paths.length > 1 ? paths.length - 1 : 1);

                var lev;
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
                    fullId = generateId(ty);

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
                    if (index !== -1 || lev <= MAX_LEV_DISTANCE) {
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
                "others": sortResults(results),
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

        function initSearchNav() {
            var hoverTimeout;

            var click_func = function(e) {
                var el = e.target;
                // to retrieve the real "owner" of the event.
                while (el.tagName !== "TR") {
                    el = el.parentNode;
                }
                var dst = e.target.getElementsByTagName("a");
                if (dst.length < 1) {
                    return;
                }
                dst = dst[0];
                if (window.location.pathname === dst.pathname) {
                    hideSearchResults();
                    document.location.href = dst.href;
                }
            };
            var mouseover_func = function(e) {
                if (mouseMovedAfterSearch) {
                    var el = e.target;
                    // to retrieve the real "owner" of the event.
                    while (el.tagName !== "TR") {
                        el = el.parentNode;
                    }
                    clearTimeout(hoverTimeout);
                    hoverTimeout = setTimeout(function() {
                        onEachLazy(document.getElementsByClassName("search-results"), function(e) {
                            onEachLazy(e.getElementsByClassName("result"), function(i_e) {
                                removeClass(i_e, "highlighted");
                            });
                        });
                        addClass(el, "highlighted");
                    }, 20);
                }
            };
            onEachLazy(document.getElementsByClassName("search-results"), function(e) {
                onEachLazy(e.getElementsByClassName("result"), function(i_e) {
                    i_e.onclick = click_func;
                    i_e.onmouseover = mouseover_func;
                });
            });

            search_input.onkeydown = function(e) {
                // "actives" references the currently highlighted item in each search tab.
                // Each array in "actives" represents a tab.
                var actives = [[], [], []];
                // "current" is used to know which tab we're looking into.
                var current = 0;
                onEachLazy(document.getElementById("results").childNodes, function(e) {
                    onEachLazy(e.getElementsByClassName("highlighted"), function(h_e) {
                        actives[current].push(h_e);
                    });
                    current += 1;
                });

                if (e.which === 38) { // up
                    if (!actives[currentTab].length ||
                        !actives[currentTab][0].previousElementSibling) {
                        return;
                    }

                    addClass(actives[currentTab][0].previousElementSibling, "highlighted");
                    removeClass(actives[currentTab][0], "highlighted");
                    e.preventDefault();
                } else if (e.which === 40) { // down
                    if (!actives[currentTab].length) {
                        var results = document.getElementById("results").childNodes;
                        if (results.length > 0) {
                            var res = results[currentTab].getElementsByClassName("result");
                            if (res.length > 0) {
                                addClass(res[0], "highlighted");
                            }
                        }
                    } else if (actives[currentTab][0].nextElementSibling) {
                        addClass(actives[currentTab][0].nextElementSibling, "highlighted");
                        removeClass(actives[currentTab][0], "highlighted");
                    }
                    e.preventDefault();
                } else if (e.which === 13) { // return
                    if (actives[currentTab].length) {
                        document.location.href =
                            actives[currentTab][0].getElementsByTagName("a")[0].href;
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
                    removeClass(actives[currentTab][0], "highlighted");
                }
            };
        }

        function buildHrefAndPath(item) {
            var displayPath;
            var href;
            var type = itemTypes[item.ty];
            var name = item.name;
            var path = item.path;

            if (type === "mod") {
                displayPath = path + "::";
                href = rootPath + path.replace(/::/g, "/") + "/" +
                       name + "/index.html";
            } else if (type === "primitive" || type === "keyword") {
                displayPath = "";
                href = rootPath + path.replace(/::/g, "/") +
                       "/" + type + "." + name + ".html";
            } else if (type === "externcrate") {
                displayPath = "";
                href = rootPath + name + "/index.html";
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
                    var splitPath = item.path.split("::");
                    var enumName = splitPath.pop();
                    path = splitPath.join("::");
                    displayPath = path + "::" + enumName + "::" + myparent.name + "::";
                    anchor = "#variant." + myparent.name + ".field." + name;
                    pageType = "enum";
                    pageName = enumName;
                } else {
                    displayPath = path + "::" + myparent.name + "::";
                }
                href = rootPath + path.replace(/::/g, "/") +
                       "/" + pageType +
                       "." + pageName +
                       ".html" + anchor;
            } else {
                displayPath = item.path + "::";
                href = rootPath + item.path.replace(/::/g, "/") +
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
            var extraStyle = "";
            if (display === false) {
                extraStyle = " style=\"display: none;\"";
            }

            var output = "";
            var duplicates = {};
            var length = 0;
            if (array.length > 0) {
                output = "<table class=\"search-results\"" + extraStyle + ">";

                array.forEach(function(item) {
                    var name, type;

                    name = item.name;
                    type = itemTypes[item.ty];

                    if (item.is_alias !== true) {
                        if (duplicates[item.fullPath]) {
                            return;
                        }
                        duplicates[item.fullPath] = true;
                    }
                    length += 1;

                    output += "<tr class=\"" + type + " result\"><td>" +
                              "<a href=\"" + item.href + "\">" +
                              (item.is_alias === true ?
                               ("<span class=\"alias\"><b>" + item.alias + " </b></span><span " +
                                  "class=\"grey\"><i>&nbsp;- see&nbsp;</i></span>") : "") +
                              item.displayPath + "<span class=\"" + type + "\">" +
                              name + "</span></a></td><td>" +
                              "<a href=\"" + item.href + "\">" +
                              "<span class=\"desc\">" + escape(item.desc) +
                              "&nbsp;</span></a></td></tr>";
                });
                output += "</table>";
            } else {
                output = "<div class=\"search-failed\"" + extraStyle + ">No results :(<br/>" +
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
                    " <a href=\"https://crates.io/\">crates.io</a>.</li></ul></div>";
            }
            return [output, length];
        }

        function makeTabHeader(tabNb, text, nbElems) {
            if (currentTab === tabNb) {
                return "<div class=\"selected\">" + text +
                       " <div class=\"count\">(" + nbElems + ")</div></div>";
            }
            return "<div>" + text + " <div class=\"count\">(" + nbElems + ")</div></div>";
        }

        function showResults(results) {
            var search = getSearchElement();
            if (results.others.length === 1
                && getSettingValue("go-to-only-result") === "true"
                // By default, the search DOM element is "empty" (meaning it has no children not
                // text content). Once a search has been run, it won't be empty, even if you press
                // ESC or empty the search input (which also "cancels" the search).
                && (!search.firstChild || search.firstChild.innerText !== getSearchLoadingText()))
            {
                var elem = document.createElement("a");
                elem.href = results.others[0].href;
                elem.style.display = "none";
                // For firefox, we need the element to be in the DOM so it can be clicked.
                document.body.appendChild(elem);
                elem.click();
                return;
            }
            var query = getQuery(search_input.value);

            currentResults = query.id;

            var ret_others = addTab(results.others, query);
            var ret_in_args = addTab(results.in_args, query, false);
            var ret_returned = addTab(results.returned, query, false);

            var output = "<h1>Results for " + escape(query.query) +
                (query.type ? " (type: " + escape(query.type) + ")" : "") + "</h1>" +
                "<div id=\"titles\">" +
                makeTabHeader(0, "In Names", ret_others[1]) +
                makeTabHeader(1, "In Parameters", ret_in_args[1]) +
                makeTabHeader(2, "In Return Types", ret_returned[1]) +
                "</div><div id=\"results\">" +
                ret_others[0] + ret_in_args[0] + ret_returned[0] + "</div>";

            search.innerHTML = output;
            showSearchResults(search);
            var tds = search.getElementsByTagName("td");
            var td_width = 0;
            if (tds.length > 0) {
                td_width = tds[0].offsetWidth;
            }
            var width = search.offsetWidth - 40 - td_width;
            onEachLazy(search.getElementsByClassName("desc"), function(e) {
                e.style.width = width + "px";
            });
            initSearchNav();
            var elems = document.getElementById("titles").childNodes;
            elems[0].onclick = function() { printTab(0); };
            elems[1].onclick = function() { printTab(1); };
            elems[2].onclick = function() { printTab(2); };
            printTab(currentTab);
        }

        function execSearch(query, searchWords, filterCrates) {
            function getSmallest(arrays, positions, notDuplicates) {
                var start = null;

                for (var it = 0; it < positions.length; ++it) {
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

                for (var x = 0; x < arrays.length; ++x) {
                    positions.push(0);
                }
                while (ret.length < MAX_RESULTS) {
                    var smallest = getSmallest(arrays, positions, notDuplicates);

                    if (smallest === null) {
                        break;
                    }
                    for (x = 0; x < arrays.length && ret.length < MAX_RESULTS; ++x) {
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

            var queries = query.raw.split(",");
            var results = {
                "in_args": [],
                "returned": [],
                "others": [],
            };

            for (var i = 0; i < queries.length; ++i) {
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

            if (elem && elem.value !== "All crates" && hasOwnProperty(rawSearchIndex, elem.value)) {
                return elem.value;
            }
            return undefined;
        }

        function search(e, forced) {
            var params = getQueryStringParams();
            var query = getQuery(search_input.value.trim());

            if (e) {
                e.preventDefault();
            }

            if (query.query.length === 0) {
                return;
            }
            if (forced !== true && query.id === currentResults) {
                if (query.query.length > 0) {
                    putBackSearch(search_input);
                }
                return;
            }

            // Update document title to maintain a meaningful browser history
            searchTitle = "Results for " + query.query + " - Rust";

            // Because searching is incremental by character, only the most
            // recent search query is added to the browser history.
            if (browserSupportsHistoryApi()) {
                if (!history.state && !params.search) {
                    history.pushState(query, "", "?search=" + encodeURIComponent(query.raw));
                } else {
                    history.replaceState(query, "", "?search=" + encodeURIComponent(query.raw));
                }
            }

            var filterCrates = getFilterCrates();
            showResults(execSearch(query, index, filterCrates));
        }

        function buildIndex(rawSearchIndex) {
            searchIndex = [];
            var searchWords = [];
            var i;
            var currentIndex = 0;

            for (var crate in rawSearchIndex) {
                if (!hasOwnProperty(rawSearchIndex, crate)) { continue; }

                var crateSize = 0;

                searchWords.push(crate);
                searchIndex.push({
                    crate: crate,
                    ty: 1, // == ExternCrate
                    name: crate,
                    path: "",
                    desc: rawSearchIndex[crate].doc,
                    type: null,
                });
                currentIndex += 1;

                // an array of [(Number) item type,
                //              (String) name,
                //              (String) full path or empty string for previous path,
                //              (String) description,
                //              (Number | null) the parent path index to `paths`]
                //              (Object | null) the type of the function (if any)
                var items = rawSearchIndex[crate].i;
                // an array of [(Number) item type,
                //              (String) name]
                var paths = rawSearchIndex[crate].p;
                // a array of [(String) alias name
                //             [Number] index to items]
                var aliases = rawSearchIndex[crate].a;

                // convert `rawPaths` entries into object form
                var len = paths.length;
                for (i = 0; i < len; ++i) {
                    paths[i] = {ty: paths[i][0], name: paths[i][1]};
                }

                // convert `items` into an object form, and construct word indices.
                //
                // before any analysis is performed lets gather the search terms to
                // search against apart from the rest of the data.  This is a quick
                // operation that is cached for the life of the page state so that
                // all other search operations have access to this cached data for
                // faster analysis operations
                len = items.length;
                var lastPath = "";
                for (i = 0; i < len; ++i) {
                    var rawRow = items[i];
                    if (!rawRow[2]) {
                        rawRow[2] = lastPath;
                    }
                    var row = {
                        crate: crate,
                        ty: rawRow[0],
                        name: rawRow[1],
                        path: rawRow[2],
                        desc: rawRow[3],
                        parent: paths[rawRow[4]],
                        type: rawRow[5],
                    };
                    searchIndex.push(row);
                    if (typeof row.name === "string") {
                        var word = row.name.toLowerCase();
                        searchWords.push(word);
                    } else {
                        searchWords.push("");
                    }
                    lastPath = row.path;
                    crateSize += 1;
                }

                if (aliases) {
                    ALIASES[crate] = {};
                    var j, local_aliases;
                    for (var alias_name in aliases) {
                        if (!aliases.hasOwnProperty(alias_name)) { continue; }

                        if (!ALIASES[crate].hasOwnProperty(alias_name)) {
                            ALIASES[crate][alias_name] = [];
                        }
                        local_aliases = aliases[alias_name];
                        for (j = 0; j < local_aliases.length; ++j) {
                            ALIASES[crate][alias_name].push(local_aliases[j] + currentIndex);
                        }
                    }
                }
                currentIndex += crateSize;
            }
            return searchWords;
        }

        function startSearch() {
            var callback = function() {
                clearInputTimeout();
                if (search_input.value.length === 0) {
                    if (browserSupportsHistoryApi()) {
                        history.replaceState("", window.currentCrate + " - Rust", "?search=");
                    }
                    hideSearchResults();
                } else {
                    searchTimeout = setTimeout(search, 500);
                }
            };
            search_input.onkeyup = callback;
            search_input.oninput = callback;
            document.getElementsByClassName("search-form")[0].onsubmit = function(e) {
                e.preventDefault();
                clearInputTimeout();
                search();
            };
            search_input.onchange = function(e) {
                if (e.target !== document.activeElement) {
                    // To prevent doing anything when it's from a blur event.
                    return;
                }
                // Do NOT e.preventDefault() here. It will prevent pasting.
                clearInputTimeout();
                // zero-timeout necessary here because at the time of event handler execution the
                // pasted content is not in the input field yet. Shouldn’t make any difference for
                // change, though.
                setTimeout(search, 0);
            };
            search_input.onpaste = search_input.onchange;

            var selectCrate = document.getElementById("crate-search");
            if (selectCrate) {
                selectCrate.onchange = function() {
                    updateLocalStorage("rustdoc-saved-filter-crate", selectCrate.value);
                    search(undefined, true);
                };
            }

            // Push and pop states are used to add search results to the browser
            // history.
            if (browserSupportsHistoryApi()) {
                // Store the previous <title> so we can revert back to it later.
                var previousTitle = document.title;

                window.addEventListener("popstate", function(e) {
                    var params = getQueryStringParams();
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
                        search_input.value = params.search;
                        // Some browsers fire "onpopstate" for every page load
                        // (Chrome), while others fire the event only when actually
                        // popping a state (Firefox), which is why search() is
                        // called both here and at the end of the startSearch()
                        // function.
                        search(e);
                    } else {
                        search_input.value = "";
                        // When browsing back from search results the main page
                        // visibility must be reset.
                        hideSearchResults();
                    }
                });
            }
            search();
        }

        index = buildIndex(rawSearchIndex);
        startSearch();

        // Draw a convenient sidebar of known crates if we have a listing
        if (rootPath === "../" || rootPath === "./") {
            var sidebar = document.getElementsByClassName("sidebar-elems")[0];
            if (sidebar) {
                var div = document.createElement("div");
                div.className = "block crate";
                div.innerHTML = "<h3>Crates</h3>";
                var ul = document.createElement("ul");
                div.appendChild(ul);

                var crates = [];
                for (var crate in rawSearchIndex) {
                    if (!hasOwnProperty(rawSearchIndex, crate)) {
                        continue;
                    }
                    crates.push(crate);
                }
                crates.sort();
                for (var i = 0; i < crates.length; ++i) {
                    var klass = "crate";
                    if (rootPath !== "./" && crates[i] === window.currentCrate) {
                        klass += " current";
                    }
                    var link = document.createElement("a");
                    link.href = rootPath + crates[i] + "/index.html";
                    link.title = rawSearchIndex[crates[i]].doc;
                    link.className = klass;
                    link.textContent = crates[i];

                    var li = document.createElement("li");
                    li.appendChild(link);
                    ul.appendChild(li);
                }
                sidebar.appendChild(div);
            }
        }
    };


    // delayed sidebar rendering.
    window.initSidebarItems = function(items) {
        var sidebar = document.getElementsByClassName("sidebar-elems")[0];
        var current = window.sidebarCurrent;

        function block(shortty, longty) {
            var filtered = items[shortty];
            if (!filtered) {
                return;
            }

            var div = document.createElement("div");
            div.className = "block " + shortty;
            var h3 = document.createElement("h3");
            h3.textContent = longty;
            div.appendChild(h3);
            var ul = document.createElement("ul");

            var length = filtered.length;
            for (var i = 0; i < length; ++i) {
                var item = filtered[i];
                var name = item[0];
                var desc = item[1]; // can be null

                var klass = shortty;
                if (name === current.name && shortty === current.ty) {
                    klass += " current";
                }
                var path;
                if (shortty === "mod") {
                    path = name + "/index.html";
                } else {
                    path = shortty + "." + name + ".html";
                }
                var link = document.createElement("a");
                link.href = current.relpath + path;
                link.title = desc;
                link.className = klass;
                link.textContent = name;
                var li = document.createElement("li");
                li.appendChild(link);
                ul.appendChild(li);
            }
            div.appendChild(ul);
            if (sidebar) {
                sidebar.appendChild(div);
            }
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
        block("keyword", "Keywords");
        block("traitalias", "Trait Aliases");
    };

    window.register_implementors = function(imp) {
        var implementors = document.getElementById("implementors-list");
        var synthetic_implementors = document.getElementById("synthetic-implementors-list");

        if (synthetic_implementors) {
            // This `inlined_types` variable is used to avoid having the same implementation
            // showing up twice. For example "String" in the "Sync" doc page.
            //
            // By the way, this is only used by and useful for traits implemented automatically
            // (like "Send" and "Sync").
            var inlined_types = new Set();
            onEachLazy(synthetic_implementors.getElementsByClassName("impl"), function(el) {
                var aliases = el.getAttribute("aliases");
                if (!aliases) {
                    return;
                }
                aliases.split(",").forEach(function(alias) {
                    inlined_types.add(alias);
                });
            });
        }

        var libs = Object.getOwnPropertyNames(imp);
        var llength = libs.length;
        for (var i = 0; i < llength; ++i) {
            if (libs[i] === currentCrate) { continue; }
            var structs = imp[libs[i]];

            var slength = structs.length;
            struct_loop:
            for (var j = 0; j < slength; ++j) {
                var struct = structs[j];

                var list = struct.synthetic ? synthetic_implementors : implementors;

                if (struct.synthetic) {
                    var stlength = struct.types.length;
                    for (var k = 0; k < stlength; k++) {
                        if (inlined_types.has(struct.types[k])) {
                            continue struct_loop;
                        }
                        inlined_types.add(struct.types[k]);
                    }
                }

                var code = document.createElement("code");
                code.innerHTML = struct.text;

                var x = code.getElementsByTagName("a");
                var xlength = x.length;
                for (var it = 0; it < xlength; it++) {
                    var href = x[it].getAttribute("href");
                    if (href && href.indexOf("http") !== 0) {
                        x[it].setAttribute("href", rootPath + href);
                    }
                }
                var display = document.createElement("h3");
                addClass(display, "impl");
                display.innerHTML = "<span class=\"in-band\"><table class=\"table-display\">" +
                    "<tbody><tr><td><code>" + code.outerHTML + "</code></td><td></td></tr>" +
                    "</tbody></table></span>";
                list.appendChild(display);
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
        return "\u2212"; // "\u2212" is "−" minus sign
    }

    function onEveryMatchingChild(elem, className, func) {
        if (elem && className && func) {
            var length = elem.childNodes.length;
            var nodes = elem.childNodes;
            for (var i = 0; i < length; ++i) {
                if (hasClass(nodes[i], className)) {
                    func(nodes[i]);
                } else {
                    onEveryMatchingChild(nodes[i], className, func);
                }
            }
        }
    }

    function toggleAllDocs(pageId, fromAutoCollapse) {
        var innerToggle = document.getElementById(toggleAllDocsId);
        if (!innerToggle) {
            return;
        }
        if (hasClass(innerToggle, "will-expand")) {
            updateLocalStorage("rustdoc-collapse", "false");
            removeClass(innerToggle, "will-expand");
            onEveryMatchingChild(innerToggle, "inner", function(e) {
                e.innerHTML = labelForToggleButton(false);
            });
            innerToggle.title = "collapse all docs";
            if (fromAutoCollapse !== true) {
                onEachLazy(document.getElementsByClassName("collapse-toggle"), function(e) {
                    collapseDocs(e, "show");
                });
            }
        } else {
            updateLocalStorage("rustdoc-collapse", "true");
            addClass(innerToggle, "will-expand");
            onEveryMatchingChild(innerToggle, "inner", function(e) {
                var parent = e.parentNode;
                var superParent = null;

                if (parent) {
                    superParent = parent.parentNode;
                }
                if (!parent || !superParent || superParent.id !== "main" ||
                    hasClass(parent, "impl") === false) {
                    e.innerHTML = labelForToggleButton(true);
                }
            });
            innerToggle.title = "expand all docs";
            if (fromAutoCollapse !== true) {
                onEachLazy(document.getElementsByClassName("collapse-toggle"), function(e) {
                    var parent = e.parentNode;
                    var superParent = null;

                    if (parent) {
                        superParent = parent.parentNode;
                    }
                    if (!parent || !superParent || superParent.id !== "main" ||
                        hasClass(parent, "impl") === false) {
                        collapseDocs(e, "hide", pageId);
                    }
                });
            }
        }
    }

    function collapseDocs(toggle, mode, pageId) {
        if (!toggle || !toggle.parentNode) {
            return;
        }

        function adjustToggle(arg) {
            return function(e) {
                if (hasClass(e, "toggle-label")) {
                    if (arg) {
                        e.style.display = "inline-block";
                    } else {
                        e.style.display = "none";
                    }
                }
                if (hasClass(e, "inner")) {
                    e.innerHTML = labelForToggleButton(arg);
                }
            };
        }

        function implHider(addOrRemove, fullHide) {
            return function(n) {
                var is_method = hasClass(n, "method") || fullHide;
                if (is_method || hasClass(n, "type")) {
                    if (is_method === true) {
                        if (addOrRemove) {
                            addClass(n, "hidden-by-impl-hider");
                        } else {
                            removeClass(n, "hidden-by-impl-hider");
                        }
                    }
                    var ns = n.nextElementSibling;
                    while (ns && (hasClass(ns, "docblock") || hasClass(ns, "item-info"))) {
                        if (addOrRemove) {
                            addClass(ns, "hidden-by-impl-hider");
                        } else {
                            removeClass(ns, "hidden-by-impl-hider");
                        }
                        ns = ns.nextElementSibling;
                    }
                }
            };
        }

        var relatedDoc;
        var action = mode;
        if (hasClass(toggle.parentNode, "impl") === false) {
            relatedDoc = toggle.parentNode.nextElementSibling;
            if (hasClass(relatedDoc, "item-info")) {
                relatedDoc = relatedDoc.nextElementSibling;
            }
            if (hasClass(relatedDoc, "docblock") || hasClass(relatedDoc, "sub-variant")) {
                if (mode === "toggle") {
                    if (hasClass(relatedDoc, "hidden-by-usual-hider")) {
                        action = "show";
                    } else {
                        action = "hide";
                    }
                }
                if (action === "hide") {
                    addClass(relatedDoc, "hidden-by-usual-hider");
                    onEachLazy(toggle.childNodes, adjustToggle(true));
                    addClass(toggle.parentNode, "collapsed");
                } else if (action === "show") {
                    removeClass(relatedDoc, "hidden-by-usual-hider");
                    removeClass(toggle.parentNode, "collapsed");
                    onEachLazy(toggle.childNodes, adjustToggle(false));
                }
            }
        } else {
            // we are collapsing the impl block(s).

            var parentElem = toggle.parentNode;
            relatedDoc = parentElem;
            var docblock = relatedDoc.nextElementSibling;

            while (hasClass(relatedDoc, "impl-items") === false) {
                relatedDoc = relatedDoc.nextElementSibling;
            }

            if (!relatedDoc && hasClass(docblock, "docblock") === false) {
                return;
            }

            // Hide all functions, but not associated types/consts.

            if (mode === "toggle") {
                if (hasClass(relatedDoc, "fns-now-collapsed") ||
                    hasClass(docblock, "hidden-by-impl-hider")) {
                    action = "show";
                } else {
                    action = "hide";
                }
            }

            var dontApplyBlockRule = toggle.parentNode.parentNode.id !== "main";
            if (action === "show") {
                removeClass(relatedDoc, "fns-now-collapsed");
                // Stability/deprecation/portability information is never hidden.
                if (hasClass(docblock, "item-info") === false) {
                    removeClass(docblock, "hidden-by-usual-hider");
                }
                onEachLazy(toggle.childNodes, adjustToggle(false, dontApplyBlockRule));
                onEachLazy(relatedDoc.childNodes, implHider(false, dontApplyBlockRule));
            } else if (action === "hide") {
                addClass(relatedDoc, "fns-now-collapsed");
                // Stability/deprecation/portability information should be shown even when detailed
                // info is hidden.
                if (hasClass(docblock, "item-info") === false) {
                    addClass(docblock, "hidden-by-usual-hider");
                }
                onEachLazy(toggle.childNodes, adjustToggle(true, dontApplyBlockRule));
                onEachLazy(relatedDoc.childNodes, implHider(true, dontApplyBlockRule));
            }
        }
    }

    function collapser(pageId, e, collapse) {
        // inherent impl ids are like "impl" or impl-<number>'.
        // they will never be hidden by default.
        var n = e.parentElement;
        if (n.id.match(/^impl(?:-\d+)?$/) === null) {
            // Automatically minimize all non-inherent impls
            if (collapse || hasClass(n, "impl")) {
                collapseDocs(e, "hide", pageId);
            }
        }
    }

    function autoCollapse(pageId, collapse) {
        if (collapse) {
            toggleAllDocs(pageId, true);
        } else if (getSettingValue("auto-hide-trait-implementations") !== "false") {
            var impl_list = document.getElementById("trait-implementations-list");

            if (impl_list !== null) {
                onEachLazy(impl_list.getElementsByClassName("collapse-toggle"), function(e) {
                    collapser(pageId, e, collapse);
                });
            }

            var blanket_list = document.getElementById("blanket-implementations-list");

            if (blanket_list !== null) {
                onEachLazy(blanket_list.getElementsByClassName("collapse-toggle"), function(e) {
                    collapser(pageId, e, collapse);
                });
            }
        }
    }

    function insertAfter(newNode, referenceNode) {
        referenceNode.parentNode.insertBefore(newNode, referenceNode.nextSibling);
    }

    function createSimpleToggle(sectionIsCollapsed) {
        var toggle = document.createElement("a");
        toggle.href = "javascript:void(0)";
        toggle.className = "collapse-toggle";
        toggle.innerHTML = "[<span class=\"inner\">" + labelForToggleButton(sectionIsCollapsed) +
                           "</span>]";
        return toggle;
    }

    function createToggle(toggle, otherMessage, fontSize, extraClass, show) {
        var span = document.createElement("span");
        span.className = "toggle-label";
        if (show) {
            span.style.display = "none";
        }
        if (!otherMessage) {
            span.innerHTML = "&nbsp;Expand&nbsp;description";
        } else {
            span.innerHTML = otherMessage;
        }

        if (fontSize) {
            span.style.fontSize = fontSize;
        }

        var mainToggle = toggle.cloneNode(true);
        mainToggle.appendChild(span);

        var wrapper = document.createElement("div");
        wrapper.className = "toggle-wrapper";
        if (!show) {
            addClass(wrapper, "collapsed");
            var inner = mainToggle.getElementsByClassName("inner");
            if (inner && inner.length > 0) {
                inner[0].innerHTML = "+";
            }
        }
        if (extraClass) {
            addClass(wrapper, extraClass);
        }
        wrapper.appendChild(mainToggle);
        return wrapper;
    }

    (function() {
        var toggles = document.getElementById(toggleAllDocsId);
        if (toggles) {
            toggles.onclick = toggleAllDocs;
        }

        var toggle = createSimpleToggle(false);
        var hideMethodDocs = getSettingValue("auto-hide-method-docs") === "true";
        var hideImplementors = getSettingValue("auto-collapse-implementors") !== "false";
        var pageId = getPageId();

        var func = function(e) {
            var next = e.nextElementSibling;
            if (next && hasClass(next, "item-info")) {
              next = next.nextElementSibling;
            }
            if (!next) {
                return;
            }
            if (hasClass(next, "docblock")) {
                var newToggle = toggle.cloneNode(true);
                insertAfter(newToggle, e.childNodes[e.childNodes.length - 1]);
                if (hideMethodDocs === true && hasClass(e, "method") === true) {
                    collapseDocs(newToggle, "hide", pageId);
                }
            }
        };

        var funcImpl = function(e) {
            var next = e.nextElementSibling;
            if (next && hasClass(next, "item-info")) {
                next = next.nextElementSibling;
            }
            if (next && hasClass(next, "docblock")) {
                next = next.nextElementSibling;
            }
            if (!next) {
                return;
            }
            if (hasClass(e, "impl") &&
                (next.getElementsByClassName("method").length > 0 ||
                 next.getElementsByClassName("associatedconstant").length > 0)) {
                var newToggle = toggle.cloneNode(true);
                insertAfter(newToggle, e.childNodes[e.childNodes.length - 1]);
                // In case the option "auto-collapse implementors" is not set to false, we collapse
                // all implementors.
                if (hideImplementors === true && e.parentNode.id === "implementors-list") {
                    collapseDocs(newToggle, "hide", pageId);
                }
            }
        };

        onEachLazy(document.getElementsByClassName("method"), func);
        onEachLazy(document.getElementsByClassName("associatedconstant"), func);
        onEachLazy(document.getElementsByClassName("impl"), funcImpl);
        var impl_call = function() {};
        if (hideMethodDocs === true) {
            impl_call = function(e, newToggle) {
                if (e.id.match(/^impl(?:-\d+)?$/) === null) {
                    // Automatically minimize all non-inherent impls
                    if (hasClass(e, "impl") === true) {
                        collapseDocs(newToggle, "hide", pageId);
                    }
                }
            };
        }
        var newToggle = document.createElement("a");
        newToggle.href = "javascript:void(0)";
        newToggle.className = "collapse-toggle hidden-default collapsed";
        newToggle.innerHTML = "[<span class=\"inner\">" + labelForToggleButton(true) +
                              "</span>] Show hidden undocumented items";
        function toggleClicked() {
            if (hasClass(this, "collapsed")) {
                removeClass(this, "collapsed");
                onEachLazy(this.parentNode.getElementsByClassName("hidden"), function(x) {
                    if (hasClass(x, "content") === false) {
                        removeClass(x, "hidden");
                        addClass(x, "x");
                    }
                }, true);
                this.innerHTML = "[<span class=\"inner\">" + labelForToggleButton(false) +
                                 "</span>] Hide undocumented items";
            } else {
                addClass(this, "collapsed");
                onEachLazy(this.parentNode.getElementsByClassName("x"), function(x) {
                    if (hasClass(x, "content") === false) {
                        addClass(x, "hidden");
                        removeClass(x, "x");
                    }
                }, true);
                this.innerHTML = "[<span class=\"inner\">" + labelForToggleButton(true) +
                                 "</span>] Show hidden undocumented items";
            }
        }
        onEachLazy(document.getElementsByClassName("impl-items"), function(e) {
            onEachLazy(e.getElementsByClassName("associatedconstant"), func);
            var hiddenElems = e.getElementsByClassName("hidden");
            var needToggle = false;

            var hlength = hiddenElems.length;
            for (var i = 0; i < hlength; ++i) {
                if (hasClass(hiddenElems[i], "content") === false &&
                    hasClass(hiddenElems[i], "docblock") === false) {
                    needToggle = true;
                    break;
                }
            }
            if (needToggle === true) {
                var inner_toggle = newToggle.cloneNode(true);
                inner_toggle.onclick = toggleClicked;
                e.insertBefore(inner_toggle, e.firstChild);
                impl_call(e.previousSibling, inner_toggle);
            }
        });

        var currentType = document.getElementsByClassName("type-decl")[0];
        var className = null;
        if (currentType) {
            currentType = currentType.getElementsByClassName("rust")[0];
            if (currentType) {
                currentType.classList.forEach(function(item) {
                    if (item !== "main") {
                        className = item;
                        return true;
                    }
                });
            }
        }
        var showItemDeclarations = getSettingValue("auto-hide-" + className);
        if (showItemDeclarations === null) {
            if (className === "enum" || className === "macro") {
                showItemDeclarations = "false";
            } else if (className === "struct" || className === "union" || className === "trait") {
                showItemDeclarations = "true";
            } else {
                // In case we found an unknown type, we just use the "parent" value.
                showItemDeclarations = getSettingValue("auto-hide-declarations");
            }
        }
        showItemDeclarations = showItemDeclarations === "false";
        function buildToggleWrapper(e) {
            if (hasClass(e, "autohide")) {
                var wrap = e.previousElementSibling;
                if (wrap && hasClass(wrap, "toggle-wrapper")) {
                    var inner_toggle = wrap.childNodes[0];
                    var extra = e.childNodes[0].tagName === "H3";

                    e.style.display = "none";
                    addClass(wrap, "collapsed");
                    onEachLazy(inner_toggle.getElementsByClassName("inner"), function(e) {
                        e.innerHTML = labelForToggleButton(true);
                    });
                    onEachLazy(inner_toggle.getElementsByClassName("toggle-label"), function(e) {
                        e.style.display = "inline-block";
                        if (extra === true) {
                            e.innerHTML = " Show " + e.childNodes[0].innerHTML;
                        }
                    });
                }
            }
            if (e.parentNode.id === "main") {
                var otherMessage = "";
                var fontSize;
                var extraClass;

                if (hasClass(e, "type-decl")) {
                    fontSize = "20px";
                    otherMessage = "&nbsp;Show&nbsp;declaration";
                    if (showItemDeclarations === false) {
                        extraClass = "collapsed";
                    }
                } else if (hasClass(e, "sub-variant")) {
                    otherMessage = "&nbsp;Show&nbsp;fields";
                } else if (hasClass(e, "non-exhaustive")) {
                    otherMessage = "&nbsp;This&nbsp;";
                    if (hasClass(e, "non-exhaustive-struct")) {
                        otherMessage += "struct";
                    } else if (hasClass(e, "non-exhaustive-enum")) {
                        otherMessage += "enum";
                    } else if (hasClass(e, "non-exhaustive-variant")) {
                        otherMessage += "enum variant";
                    } else if (hasClass(e, "non-exhaustive-type")) {
                        otherMessage += "type";
                    }
                    otherMessage += "&nbsp;is&nbsp;marked&nbsp;as&nbsp;non-exhaustive";
                } else if (hasClass(e.childNodes[0], "impl-items")) {
                    extraClass = "marg-left";
                }

                e.parentNode.insertBefore(
                    createToggle(
                        toggle,
                        otherMessage,
                        fontSize,
                        extraClass,
                        hasClass(e, "type-decl") === false || showItemDeclarations === true),
                    e);
                if (hasClass(e, "type-decl") === true && showItemDeclarations === true) {
                    collapseDocs(e.previousSibling.childNodes[0], "toggle");
                }
                if (hasClass(e, "non-exhaustive") === true) {
                    collapseDocs(e.previousSibling.childNodes[0], "toggle");
                }
            }
        }

        onEachLazy(document.getElementsByClassName("docblock"), buildToggleWrapper);
        onEachLazy(document.getElementsByClassName("sub-variant"), buildToggleWrapper);
        var pageId = getPageId();

        autoCollapse(pageId, getSettingValue("collapse") === "true");

        if (pageId !== null) {
            expandSection(pageId);
        }
    }());

    function createToggleWrapper(tog) {
        var span = document.createElement("span");
        span.className = "toggle-label";
        span.style.display = "none";
        span.innerHTML = "&nbsp;Expand&nbsp;attributes";
        tog.appendChild(span);

        var wrapper = document.createElement("div");
        wrapper.className = "toggle-wrapper toggle-attributes";
        wrapper.appendChild(tog);
        return wrapper;
    }

    (function() {
        // To avoid checking on "rustdoc-item-attributes" value on every loop...
        var itemAttributesFunc = function() {};
        if (getSettingValue("auto-hide-attributes") !== "false") {
            itemAttributesFunc = function(x) {
                collapseDocs(x.previousSibling.childNodes[0], "toggle");
            };
        }
        var attributesToggle = createToggleWrapper(createSimpleToggle(false));
        onEachLazy(main.getElementsByClassName("attributes"), function(i_e) {
            var attr_tog = attributesToggle.cloneNode(true);
            if (hasClass(i_e, "top-attr") === true) {
                addClass(attr_tog, "top-attr");
            }
            i_e.parentNode.insertBefore(attr_tog, i_e);
            itemAttributesFunc(i_e);
        });
    }());

    (function() {
        // To avoid checking on "rustdoc-line-numbers" value on every loop...
        var lineNumbersFunc = function() {};
        if (getSettingValue("line-numbers") === "true") {
            lineNumbersFunc = function(x) {
                var count = x.textContent.split("\n").length;
                var elems = [];
                for (var i = 0; i < count; ++i) {
                    elems.push(i + 1);
                }
                var node = document.createElement("pre");
                addClass(node, "line-number");
                node.innerHTML = elems.join("\n");
                x.parentNode.insertBefore(node, x);
            };
        }
        onEachLazy(document.getElementsByClassName("rust-example-rendered"), function(e) {
            if (hasClass(e, "compile_fail")) {
                e.addEventListener("mouseover", function() {
                    this.parentElement.previousElementSibling.childNodes[0].style.color = "#f00";
                });
                e.addEventListener("mouseout", function() {
                    this.parentElement.previousElementSibling.childNodes[0].style.color = "";
                });
            } else if (hasClass(e, "ignore")) {
                e.addEventListener("mouseover", function() {
                    this.parentElement.previousElementSibling.childNodes[0].style.color = "#ff9200";
                });
                e.addEventListener("mouseout", function() {
                    this.parentElement.previousElementSibling.childNodes[0].style.color = "";
                });
            }
            lineNumbersFunc(e);
        });
    }());

    onEachLazy(document.getElementsByClassName("notable-traits"), function(e) {
        e.onclick = function() {
            this.getElementsByClassName('notable-traits-tooltiptext')[0]
                .classList.toggle("force-tooltip");
        };
    });

    // In the search display, allows to switch between tabs.
    function printTab(nb) {
        if (nb === 0 || nb === 1 || nb === 2) {
            currentTab = nb;
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
                elem.style.display = "";
            } else {
                elem.style.display = "none";
            }
            nb -= 1;
        });
    }

    function putBackSearch(search_input) {
        var search = getSearchElement();
        if (search_input.value !== "" && hasClass(search, "hidden")) {
            showSearchResults(search);
            if (browserSupportsHistoryApi()) {
                history.replaceState(search_input.value,
                                     "",
                                     "?search=" + encodeURIComponent(search_input.value));
            }
            document.title = searchTitle;
        }
    }

    function getSearchLoadingText() {
        return "Loading search results...";
    }

    if (search_input) {
        search_input.onfocus = function() {
            putBackSearch(this);
        };
    }

    var params = getQueryStringParams();
    if (params && params.search) {
        var search = getSearchElement();
        search.innerHTML = "<h3 style=\"text-align: center;\">" + getSearchLoadingText() + "</h3>";
        showSearchResults(search);
    }

    var sidebar_menu = document.getElementsByClassName("sidebar-menu")[0];
    if (sidebar_menu) {
        sidebar_menu.onclick = function() {
            var sidebar = document.getElementsByClassName("sidebar")[0];
            if (hasClass(sidebar, "mobile") === true) {
                hideSidebar();
            } else {
                showSidebar();
            }
        };
    }

    if (main) {
        onEachLazy(main.getElementsByClassName("loading-content"), function(e) {
            e.remove();
        });
        onEachLazy(main.childNodes, function(e) {
            // Unhide the actual content once loading is complete. Headers get
            // flex treatment for their horizontal layout, divs get block treatment
            // for vertical layout (column-oriented flex layout for divs caused
            // errors in mobile browsers).
            if (e.tagName === "H2" || e.tagName === "H3") {
                var nextTagName = e.nextElementSibling.tagName;
                if (nextTagName == "H2" || nextTagName == "H3") {
                    e.nextElementSibling.style.display = "flex";
                } else {
                    e.nextElementSibling.style.display = "block";
                }
            }
        });
    }

    function enableSearchInput() {
        if (search_input) {
            search_input.removeAttribute('disabled');
        }
    }

    window.addSearchOptions = function(crates) {
        var elem = document.getElementById("crate-search");

        if (!elem) {
            enableSearchInput();
            return;
        }
        var crates_text = [];
        if (Object.keys(crates).length > 1) {
            for (var crate in crates) {
                if (hasOwnProperty(crates, crate)) {
                    crates_text.push(crate);
                }
            }
        }
        crates_text.sort(function(a, b) {
            var lower_a = a.toLowerCase();
            var lower_b = b.toLowerCase();

            if (lower_a < lower_b) {
                return -1;
            } else if (lower_a > lower_b) {
                return 1;
            }
            return 0;
        });
        var savedCrate = getSettingValue("saved-filter-crate");
        for (var i = 0; i < crates_text.length; ++i) {
            var option = document.createElement("option");
            option.value = crates_text[i];
            option.innerText = crates_text[i];
            elem.appendChild(option);
            // Set the crate filter from saved storage, if the current page has the saved crate
            // filter.
            //
            // If not, ignore the crate filter -- we want to support filtering for crates on sites
            // like doc.rust-lang.org where the crates may differ from page to page while on the
            // same domain.
            if (crates_text[i] === savedCrate) {
                elem.value = savedCrate;
            }
        }
        enableSearchInput();
    };

    function buildHelperPopup() {
        var popup = document.createElement("aside");
        addClass(popup, "hidden");
        popup.id = "help";

        var book_info = document.createElement("span");
        book_info.innerHTML = "You can find more information in \
            <a href=\"https://doc.rust-lang.org/rustdoc/\">the rustdoc book</a>.";

        var container = document.createElement("div");
        var shortcuts = [
            ["?", "Show this help dialog"],
            ["S", "Focus the search field"],
            ["T", "Focus the theme picker menu"],
            ["↑", "Move up in search results"],
            ["↓", "Move down in search results"],
            ["↹", "Switch tab"],
            ["&#9166;", "Go to active search result"],
            ["+", "Expand all sections"],
            ["-", "Collapse all sections"],
        ].map(x => "<dt><kbd>" + x[0] + "</kbd></dt><dd>" + x[1] + "</dd>").join("");
        var div_shortcuts = document.createElement("div");
        addClass(div_shortcuts, "shortcuts");
        div_shortcuts.innerHTML = "<h2>Keyboard Shortcuts</h2><dl>" + shortcuts + "</dl></div>";

        var infos = [
            "Prefix searches with a type followed by a colon (e.g., <code>fn:</code>) to \
             restrict the search to a given item kind.",
            "Accepted kinds are: <code>fn</code>, <code>mod</code>, <code>struct</code>, \
             <code>enum</code>, <code>trait</code>, <code>type</code>, <code>macro</code>, \
             and <code>const</code>.",
            "Search functions by type signature (e.g., <code>vec -&gt; usize</code> or \
             <code>* -&gt; vec</code>)",
            "Search multiple things at once by splitting your query with comma (e.g., \
             <code>str,u8</code> or <code>String,struct:Vec,test</code>)",
            "You can look for items with an exact name by putting double quotes around \
             your request: <code>\"string\"</code>",
            "Look for items inside another one by searching for a path: <code>vec::Vec</code>",
        ].map(x => "<p>" + x + "</p>").join("");
        var div_infos = document.createElement("div");
        addClass(div_infos, "infos");
        div_infos.innerHTML = "<h2>Search Tricks</h2>" + infos;

        container.appendChild(book_info);
        container.appendChild(div_shortcuts);
        container.appendChild(div_infos);

        popup.appendChild(container);
        insertAfter(popup, getSearchElement());
        // So that it's only built once and then it'll do nothing when called!
        buildHelperPopup = function() {};
    }

    onHashChange(null);
    window.onhashchange = onHashChange;
}());

// This is required in firefox. Explanations: when going back in the history, firefox doesn't re-run
// the JS, therefore preventing rustdoc from setting a few things required to be able to reload the
// previous search results (if you navigated to a search result with the keyboard, pressed enter on
// it to navigate to that result, and then came back to this page).
window.onunload = function(){};
