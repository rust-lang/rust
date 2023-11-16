/* global addClass, getNakedUrl, getSettingValue */
/* global onEachLazy, removeClass, searchState, browserSupportsHistoryApi, exports */

"use strict";

// polyfill
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/toSpliced
if (!Array.prototype.toSpliced) {
    // Can't use arrow functions, because we want `this`
    Array.prototype.toSpliced = function() {
        const me = this.slice();
        Array.prototype.splice.apply(me, arguments);
        return me;
    };
}

(function() {
// This mapping table should match the discriminants of
// `rustdoc::formats::item_type::ItemType` type in Rust.
const itemTypes = [
    "mod",
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
    "traitalias",
    "generic",
];

const longItemTypes = [
    "module",
    "extern crate",
    "re-export",
    "struct",
    "enum",
    "function",
    "type alias",
    "static",
    "trait",
    "",
    "trait method",
    "method",
    "struct field",
    "enum variant",
    "macro",
    "primitive type",
    "assoc type",
    "constant",
    "assoc const",
    "union",
    "foreign type",
    "keyword",
    "existential type",
    "attribute macro",
    "derive macro",
    "trait alias",
];

// used for special search precedence
const TY_PRIMITIVE = itemTypes.indexOf("primitive");
const TY_KEYWORD = itemTypes.indexOf("keyword");
const TY_GENERIC = itemTypes.indexOf("generic");
const ROOT_PATH = typeof window !== "undefined" ? window.rootPath : "../";

function hasOwnPropertyRustdoc(obj, property) {
    return Object.prototype.hasOwnProperty.call(obj, property);
}

// In the search display, allows to switch between tabs.
function printTab(nb) {
    let iter = 0;
    let foundCurrentTab = false;
    let foundCurrentResultSet = false;
    onEachLazy(document.getElementById("search-tabs").childNodes, elem => {
        if (nb === iter) {
            addClass(elem, "selected");
            foundCurrentTab = true;
        } else {
            removeClass(elem, "selected");
        }
        iter += 1;
    });
    const isTypeSearch = (nb > 0 || iter === 1);
    iter = 0;
    onEachLazy(document.getElementById("results").childNodes, elem => {
        if (nb === iter) {
            addClass(elem, "active");
            foundCurrentResultSet = true;
        } else {
            removeClass(elem, "active");
        }
        iter += 1;
    });
    if (foundCurrentTab && foundCurrentResultSet) {
        searchState.currentTab = nb;
        // Corrections only kick in on type-based searches.
        const correctionsElem = document.getElementsByClassName("search-corrections");
        if (isTypeSearch) {
            removeClass(correctionsElem[0], "hidden");
        } else {
            addClass(correctionsElem[0], "hidden");
        }
    } else if (nb !== 0) {
        printTab(0);
    }
}

/**
 * The [edit distance] is a metric for measuring the difference between two strings.
 *
 * [edit distance]: https://en.wikipedia.org/wiki/Edit_distance
 */

/*
 * This function was translated, mostly line-for-line, from
 * https://github.com/rust-lang/rust/blob/ff4b772f805ec1e/compiler/rustc_span/src/edit_distance.rs
 *
 * The current implementation is the restricted Damerau-Levenshtein algorithm. It is restricted
 * because it does not permit modifying characters that have already been transposed. The specific
 * algorithm should not matter to the caller of the methods, which is why it is not noted in the
 * documentation.
 */
const editDistanceState = {
    current: [],
    prev: [],
    prevPrev: [],
    calculate: function calculate(a, b, limit) {
        // Ensure that `b` is the shorter string, minimizing memory use.
        if (a.length < b.length) {
            const aTmp = a;
            a = b;
            b = aTmp;
        }

        const minDist = a.length - b.length;
        // If we know the limit will be exceeded, we can return early.
        if (minDist > limit) {
            return limit + 1;
        }

        // Strip common prefix.
        // We know that `b` is the shorter string, so we don't need to check
        // `a.length`.
        while (b.length > 0 && b[0] === a[0]) {
            a = a.substring(1);
            b = b.substring(1);
        }
        // Strip common suffix.
        while (b.length > 0 && b[b.length - 1] === a[a.length - 1]) {
            a = a.substring(0, a.length - 1);
            b = b.substring(0, b.length - 1);
        }

        // If either string is empty, the distance is the length of the other.
        // We know that `b` is the shorter string, so we don't need to check `a`.
        if (b.length === 0) {
            return minDist;
        }

        const aLength = a.length;
        const bLength = b.length;

        for (let i = 0; i <= bLength; ++i) {
            this.current[i] = 0;
            this.prev[i] = i;
            this.prevPrev[i] = Number.MAX_VALUE;
        }

        // row by row
        for (let i = 1; i <= aLength; ++i) {
            this.current[0] = i;
            const aIdx = i - 1;

            // column by column
            for (let j = 1; j <= bLength; ++j) {
                const bIdx = j - 1;

                // There is no cost to substitute a character with itself.
                const substitutionCost = a[aIdx] === b[bIdx] ? 0 : 1;

                this.current[j] = Math.min(
                    // deletion
                    this.prev[j] + 1,
                    // insertion
                    this.current[j - 1] + 1,
                    // substitution
                    this.prev[j - 1] + substitutionCost
                );

                if ((i > 1) && (j > 1) && (a[aIdx] === b[bIdx - 1]) && (a[aIdx - 1] === b[bIdx])) {
                    // transposition
                    this.current[j] = Math.min(
                        this.current[j],
                        this.prevPrev[j - 2] + 1
                    );
                }
            }

            // Rotate the buffers, reusing the memory
            const prevPrevTmp = this.prevPrev;
            this.prevPrev = this.prev;
            this.prev = this.current;
            this.current = prevPrevTmp;
        }

        // `prev` because we already rotated the buffers.
        const distance = this.prev[bLength];
        return distance <= limit ? distance : (limit + 1);
    },
};

function editDistance(a, b, limit) {
    return editDistanceState.calculate(a, b, limit);
}

function initSearch(rawSearchIndex) {
    const MAX_RESULTS = 200;
    const NO_TYPE_FILTER = -1;
    /**
     *  @type {Array<Row>}
     */
    let searchIndex;
    let currentResults;
    /**
     * Map from normalized type names to integers. Used to make type search
     * more efficient.
     *
     * @type {Map<string, integer>}
     */
    let typeNameIdMap;
    const ALIASES = new Map();

    /**
     * Special type name IDs for searching by array.
     */
    let typeNameIdOfArray;
    /**
     * Special type name IDs for searching by slice.
     */
    let typeNameIdOfSlice;
    /**
     * Special type name IDs for searching by both array and slice (`[]` syntax).
     */
    let typeNameIdOfArrayOrSlice;

    /**
     * Add an item to the type Name->ID map, or, if one already exists, use it.
     * Returns the number. If name is "" or null, return null (pure generic).
     *
     * This is effectively string interning, so that function matching can be
     * done more quickly. Two types with the same name but different item kinds
     * get the same ID.
     *
     * @param {string} name
     *
     * @returns {integer}
     */
    function buildTypeMapIndex(name) {
        if (name === "" || name === null) {
            return null;
        }

        if (typeNameIdMap.has(name)) {
            return typeNameIdMap.get(name);
        } else {
            const id = typeNameIdMap.size;
            typeNameIdMap.set(name, id);
            return id;
        }
    }

    function isWhitespace(c) {
        return " \t\n\r".indexOf(c) !== -1;
    }

    function isSpecialStartCharacter(c) {
        return "<\"".indexOf(c) !== -1;
    }

    function isEndCharacter(c) {
        return ",>-]".indexOf(c) !== -1;
    }

    function isStopCharacter(c) {
        return isEndCharacter(c);
    }

    function isErrorCharacter(c) {
        return "()".indexOf(c) !== -1;
    }

    function itemTypeFromName(typename) {
        const index = itemTypes.findIndex(i => i === typename);
        if (index < 0) {
            throw ["Unknown type filter ", typename];
        }
        return index;
    }

    /**
     * If we encounter a `"`, then we try to extract the string from it until we find another `"`.
     *
     * This function will throw an error in the following cases:
     * * There is already another string element.
     * * We are parsing a generic argument.
     * * There is more than one element.
     * * There is no closing `"`.
     *
     * @param {ParsedQuery} query
     * @param {ParserState} parserState
     * @param {boolean} isInGenerics
     */
    function getStringElem(query, parserState, isInGenerics) {
        if (isInGenerics) {
            throw ["Unexpected ", "\"", " in generics"];
        } else if (query.literalSearch) {
            throw ["Cannot have more than one literal search element"];
        } else if (parserState.totalElems - parserState.genericsElems > 0) {
            throw ["Cannot use literal search when there is more than one element"];
        }
        parserState.pos += 1;
        const start = parserState.pos;
        const end = getIdentEndPosition(parserState);
        if (parserState.pos >= parserState.length) {
            throw ["Unclosed ", "\""];
        } else if (parserState.userQuery[end] !== "\"") {
            throw ["Unexpected ", parserState.userQuery[end], " in a string element"];
        } else if (start === end) {
            throw ["Cannot have empty string element"];
        }
        // To skip the quote at the end.
        parserState.pos += 1;
        query.literalSearch = true;
    }

    /**
     * Returns `true` if the current parser position is starting with "::".
     *
     * @param {ParserState} parserState
     *
     * @return {boolean}
     */
    function isPathStart(parserState) {
        return parserState.userQuery.slice(parserState.pos, parserState.pos + 2) === "::";
    }

    /**
     * Returns `true` if the current parser position is starting with "->".
     *
     * @param {ParserState} parserState
     *
     * @return {boolean}
     */
    function isReturnArrow(parserState) {
        return parserState.userQuery.slice(parserState.pos, parserState.pos + 2) === "->";
    }

    /**
     * Returns `true` if the given `c` character is valid for an ident.
     *
     * @param {string} c
     *
     * @return {boolean}
     */
    function isIdentCharacter(c) {
        return (
            c === "_" ||
            (c >= "0" && c <= "9") ||
            (c >= "a" && c <= "z") ||
            (c >= "A" && c <= "Z"));
    }

    /**
     * Returns `true` if the given `c` character is a separator.
     *
     * @param {string} c
     *
     * @return {boolean}
     */
    function isSeparatorCharacter(c) {
        return c === ",";
    }

/**
     * Returns `true` if the given `c` character is a path separator. For example
     * `:` in `a::b` or a whitespace in `a b`.
     *
     * @param {string} c
     *
     * @return {boolean}
     */
    function isPathSeparator(c) {
        return c === ":" || isWhitespace(c);
    }

    /**
     * Returns `true` if the previous character is `lookingFor`.
     *
     * @param {ParserState} parserState
     * @param {String} lookingFor
     *
     * @return {boolean}
     */
    function prevIs(parserState, lookingFor) {
        let pos = parserState.pos;
        while (pos > 0) {
            const c = parserState.userQuery[pos - 1];
            if (c === lookingFor) {
                return true;
            } else if (!isWhitespace(c)) {
                break;
            }
            pos -= 1;
        }
        return false;
    }

    /**
     * Returns `true` if the last element in the `elems` argument has generics.
     *
     * @param {Array<QueryElement>} elems
     * @param {ParserState} parserState
     *
     * @return {boolean}
     */
    function isLastElemGeneric(elems, parserState) {
        return (elems.length > 0 && elems[elems.length - 1].generics.length > 0) ||
            prevIs(parserState, ">");
    }

    /**
     * Increase current parser position until it doesn't find a whitespace anymore.
     *
     * @param {ParserState} parserState
     */
    function skipWhitespace(parserState) {
        while (parserState.pos < parserState.userQuery.length) {
            const c = parserState.userQuery[parserState.pos];
            if (!isWhitespace(c)) {
                break;
            }
            parserState.pos += 1;
        }
    }

    /**
     * @param {ParsedQuery} query
     * @param {ParserState} parserState
     * @param {string} name                  - Name of the query element.
     * @param {Array<QueryElement>} generics - List of generics of this query element.
     *
     * @return {QueryElement}                - The newly created `QueryElement`.
     */
    function createQueryElement(query, parserState, name, generics, isInGenerics) {
        const path = name.trim();
        if (path.length === 0 && generics.length === 0) {
            throw ["Unexpected ", parserState.userQuery[parserState.pos]];
        } else if (path === "*") {
            throw ["Unexpected ", "*"];
        }
        if (query.literalSearch && parserState.totalElems - parserState.genericsElems > 0) {
            throw ["Cannot have more than one element if you use quotes"];
        }
        const typeFilter = parserState.typeFilter;
        parserState.typeFilter = null;
        if (name === "!") {
            if (typeFilter !== null && typeFilter !== "primitive") {
                throw [
                    "Invalid search type: primitive never type ",
                    "!",
                    " and ",
                    typeFilter,
                    " both specified",
                ];
            }
            if (generics.length !== 0) {
                throw [
                    "Never type ",
                    "!",
                    " does not accept generic parameters",
                ];
            }
            return {
                name: "never",
                id: null,
                fullPath: ["never"],
                pathWithoutLast: [],
                pathLast: "never",
                generics: [],
                typeFilter: "primitive",
            };
        }
        if (path.startsWith("::")) {
            throw ["Paths cannot start with ", "::"];
        } else if (path.endsWith("::")) {
            throw ["Paths cannot end with ", "::"];
        } else if (path.includes("::::")) {
            throw ["Unexpected ", "::::"];
        } else if (path.includes(" ::")) {
            throw ["Unexpected ", " ::"];
        } else if (path.includes(":: ")) {
            throw ["Unexpected ", ":: "];
        }
        const pathSegments = path.split(/::|\s+/);
        // In case we only have something like `<p>`, there is no name.
        if (pathSegments.length === 0 || (pathSegments.length === 1 && pathSegments[0] === "")) {
            if (generics.length > 0 || prevIs(parserState, ">")) {
                throw ["Found generics without a path"];
            } else {
                throw ["Unexpected ", parserState.userQuery[parserState.pos]];
            }
        }
        for (const [i, pathSegment] of pathSegments.entries()) {
            if (pathSegment === "!") {
                if (i !== 0) {
                    throw ["Never type ", "!", " is not associated item"];
                }
                pathSegments[i] = "never";
            }
        }
        parserState.totalElems += 1;
        if (isInGenerics) {
            parserState.genericsElems += 1;
        }
        return {
            name: name.trim(),
            id: null,
            fullPath: pathSegments,
            pathWithoutLast: pathSegments.slice(0, pathSegments.length - 1),
            pathLast: pathSegments[pathSegments.length - 1],
            generics: generics,
            typeFilter,
        };
    }

    /**
     * This function goes through all characters until it reaches an invalid ident character or the
     * end of the query. It returns the position of the last character of the ident.
     *
     * @param {ParserState} parserState
     *
     * @return {integer}
     */
    function getIdentEndPosition(parserState) {
        const start = parserState.pos;
        let end = parserState.pos;
        let foundExclamation = -1;
        while (parserState.pos < parserState.length) {
            const c = parserState.userQuery[parserState.pos];
            if (!isIdentCharacter(c)) {
                if (c === "!") {
                    if (foundExclamation !== -1) {
                        throw ["Cannot have more than one ", "!", " in an ident"];
                    } else if (parserState.pos + 1 < parserState.length &&
                        isIdentCharacter(parserState.userQuery[parserState.pos + 1])
                    ) {
                        throw ["Unexpected ", "!", ": it can only be at the end of an ident"];
                    }
                    foundExclamation = parserState.pos;
                } else if (isErrorCharacter(c)) {
                    throw ["Unexpected ", c];
                } else if (isPathSeparator(c)) {
                    if (c === ":") {
                        if (!isPathStart(parserState)) {
                            break;
                        }
                        // Skip current ":".
                        parserState.pos += 1;
                    } else {
                        while (parserState.pos + 1 < parserState.length) {
                            const next_c = parserState.userQuery[parserState.pos + 1];
                            if (!isWhitespace(next_c)) {
                                break;
                            }
                            parserState.pos += 1;
                        }
                    }
                    if (foundExclamation !== -1) {
                        if (foundExclamation !== start &&
                            isIdentCharacter(parserState.userQuery[foundExclamation - 1])
                        ) {
                            throw ["Cannot have associated items in macros"];
                        } else {
                            // while the never type has no associated macros, we still
                            // can parse a path like that
                            foundExclamation = -1;
                        }
                    }
                } else if (
                    c === "[" ||
                    isStopCharacter(c) ||
                    isSpecialStartCharacter(c) ||
                    isSeparatorCharacter(c)
                ) {
                    break;
                } else {
                    throw ["Unexpected ", c];
                }
            }
            parserState.pos += 1;
            end = parserState.pos;
        }
        // if start == end - 1, we got the never type
        if (foundExclamation !== -1 &&
            foundExclamation !== start &&
            isIdentCharacter(parserState.userQuery[foundExclamation - 1])
        ) {
            if (parserState.typeFilter === null) {
                parserState.typeFilter = "macro";
            } else if (parserState.typeFilter !== "macro") {
                throw [
                    "Invalid search type: macro ",
                    "!",
                    " and ",
                    parserState.typeFilter,
                    " both specified",
                ];
            }
            end = foundExclamation;
        }
        return end;
    }

    /**
     * @param {ParsedQuery} query
     * @param {ParserState} parserState
     * @param {Array<QueryElement>} elems - This is where the new {QueryElement} will be added.
     * @param {boolean} isInGenerics
     */
    function getNextElem(query, parserState, elems, isInGenerics) {
        const generics = [];

        skipWhitespace(parserState);
        let start = parserState.pos;
        let end;
        if (parserState.userQuery[parserState.pos] === "[") {
            parserState.pos += 1;
            getItemsBefore(query, parserState, generics, "]");
            const typeFilter = parserState.typeFilter;
            if (typeFilter !== null && typeFilter !== "primitive") {
                throw [
                    "Invalid search type: primitive ",
                    "[]",
                    " and ",
                    typeFilter,
                    " both specified",
                ];
            }
            parserState.typeFilter = null;
            parserState.totalElems += 1;
            if (isInGenerics) {
                parserState.genericsElems += 1;
            }
            elems.push({
                name: "[]",
                id: null,
                fullPath: ["[]"],
                pathWithoutLast: [],
                pathLast: "[]",
                generics,
                typeFilter: "primitive",
            });
        } else {
            const isStringElem = parserState.userQuery[start] === "\"";
            // We handle the strings on their own mostly to make code easier to follow.
            if (isStringElem) {
                start += 1;
                getStringElem(query, parserState, isInGenerics);
                end = parserState.pos - 1;
            } else {
                end = getIdentEndPosition(parserState);
            }
            if (parserState.pos < parserState.length &&
                parserState.userQuery[parserState.pos] === "<"
            ) {
                if (start >= end) {
                    throw ["Found generics without a path"];
                }
                parserState.pos += 1;
                getItemsBefore(query, parserState, generics, ">");
            }
            if (isStringElem) {
                skipWhitespace(parserState);
            }
            if (start >= end && generics.length === 0) {
                return;
            }
            elems.push(
                createQueryElement(
                    query,
                    parserState,
                    parserState.userQuery.slice(start, end),
                    generics,
                    isInGenerics
                )
            );
        }
    }

    /**
     * This function parses the next query element until it finds `endChar`, calling `getNextElem`
     * to collect each element.
     *
     * If there is no `endChar`, this function will implicitly stop at the end without raising an
     * error.
     *
     * @param {ParsedQuery} query
     * @param {ParserState} parserState
     * @param {Array<QueryElement>} elems - This is where the new {QueryElement} will be added.
     * @param {string} endChar            - This function will stop when it'll encounter this
     *                                      character.
     */
    function getItemsBefore(query, parserState, elems, endChar) {
        let foundStopChar = true;
        let start = parserState.pos;

        // If this is a generic, keep the outer item's type filter around.
        const oldTypeFilter = parserState.typeFilter;
        parserState.typeFilter = null;

        let extra = "";
        if (endChar === ">") {
            extra = "<";
        } else if (endChar === "]") {
            extra = "[";
        } else if (endChar === "") {
            extra = "->";
        } else {
            extra = endChar;
        }

        while (parserState.pos < parserState.length) {
            const c = parserState.userQuery[parserState.pos];
            if (c === endChar) {
                break;
            } else if (isSeparatorCharacter(c)) {
                parserState.pos += 1;
                foundStopChar = true;
                continue;
            } else if (c === ":" && isPathStart(parserState)) {
                throw ["Unexpected ", "::", ": paths cannot start with ", "::"];
            }  else if (c === ":") {
                if (parserState.typeFilter !== null) {
                    throw ["Unexpected ", ":"];
                }
                if (elems.length === 0) {
                    throw ["Expected type filter before ", ":"];
                } else if (query.literalSearch) {
                    throw ["Cannot use quotes on type filter"];
                }
                // The type filter doesn't count as an element since it's a modifier.
                const typeFilterElem = elems.pop();
                checkExtraTypeFilterCharacters(start, parserState);
                parserState.typeFilter = typeFilterElem.name;
                parserState.pos += 1;
                parserState.totalElems -= 1;
                query.literalSearch = false;
                foundStopChar = true;
                continue;
            } else if (isEndCharacter(c)) {
                throw ["Unexpected ", c, " after ", extra];
            }
            if (!foundStopChar) {
                let extra = [];
                if (isLastElemGeneric(query.elems, parserState)) {
                    extra = [" after ", ">"];
                } else if (prevIs(parserState, "\"")) {
                    throw ["Cannot have more than one element if you use quotes"];
                }
                if (endChar !== "") {
                    throw [
                        "Expected ",
                        ",",
                        " or ",
                        endChar,
                        ...extra,
                        ", found ",
                        c,
                    ];
                }
                throw [
                    "Expected ",
                    ",",
                    ...extra,
                    ", found ",
                    c,
                ];
            }
            const posBefore = parserState.pos;
            start = parserState.pos;
            getNextElem(query, parserState, elems, endChar !== "");
            if (endChar !== "" && parserState.pos >= parserState.length) {
                throw ["Unclosed ", extra];
            }
            // This case can be encountered if `getNextElem` encountered a "stop character" right
            // from the start. For example if you have `,,` or `<>`. In this case, we simply move up
            // the current position to continue the parsing.
            if (posBefore === parserState.pos) {
                parserState.pos += 1;
            }
            foundStopChar = false;
        }
        if (parserState.pos >= parserState.length && endChar !== "") {
            throw ["Unclosed ", extra];
        }
        // We are either at the end of the string or on the `endChar` character, let's move forward
        // in any case.
        parserState.pos += 1;

        parserState.typeFilter = oldTypeFilter;
    }

    /**
     * Checks that the type filter doesn't have unwanted characters like `<>` (which are ignored
     * if empty).
     *
     * @param {ParserState} parserState
     */
    function checkExtraTypeFilterCharacters(start, parserState) {
        const query = parserState.userQuery.slice(start, parserState.pos).trim();

        for (const c in query) {
            if (!isIdentCharacter(query[c])) {
                throw [
                    "Unexpected ",
                    query[c],
                    " in type filter (before ",
                    ":",
                    ")",
                ];
            }
        }
    }

    /**
     * Parses the provided `query` input to fill `parserState`. If it encounters an error while
     * parsing `query`, it'll throw an error.
     *
     * @param {ParsedQuery} query
     * @param {ParserState} parserState
     */
    function parseInput(query, parserState) {
        let foundStopChar = true;
        let start = parserState.pos;

        while (parserState.pos < parserState.length) {
            const c = parserState.userQuery[parserState.pos];
            if (isStopCharacter(c)) {
                foundStopChar = true;
                if (isSeparatorCharacter(c)) {
                    parserState.pos += 1;
                    continue;
                } else if (c === "-" || c === ">") {
                    if (isReturnArrow(parserState)) {
                        break;
                    }
                    throw ["Unexpected ", c, " (did you mean ", "->", "?)"];
                }
                throw ["Unexpected ", c];
            } else if (c === ":" && !isPathStart(parserState)) {
                if (parserState.typeFilter !== null) {
                    throw [
                        "Unexpected ",
                        ":",
                        " (expected path after type filter ",
                        parserState.typeFilter + ":",
                        ")",
                    ];
                } else if (query.elems.length === 0) {
                    throw ["Expected type filter before ", ":"];
                } else if (query.literalSearch) {
                    throw ["Cannot use quotes on type filter"];
                }
                // The type filter doesn't count as an element since it's a modifier.
                const typeFilterElem = query.elems.pop();
                checkExtraTypeFilterCharacters(start, parserState);
                parserState.typeFilter = typeFilterElem.name;
                parserState.pos += 1;
                parserState.totalElems -= 1;
                query.literalSearch = false;
                foundStopChar = true;
                continue;
            } else if (isWhitespace(c)) {
                skipWhitespace(parserState);
                continue;
            }
            if (!foundStopChar) {
                let extra = "";
                if (isLastElemGeneric(query.elems, parserState)) {
                    extra = [" after ", ">"];
                } else if (prevIs(parserState, "\"")) {
                    throw ["Cannot have more than one element if you use quotes"];
                }
                if (parserState.typeFilter !== null) {
                    throw [
                        "Expected ",
                        ",",
                        " or ",
                        "->",
                        ...extra,
                        ", found ",
                        c,
                    ];
                }
                throw [
                    "Expected ",
                    ",",
                    ", ",
                    ":",
                    " or ",
                    "->",
                    ...extra,
                    ", found ",
                    c,
                ];
            }
            const before = query.elems.length;
            start = parserState.pos;
            getNextElem(query, parserState, query.elems, false);
            if (query.elems.length === before) {
                // Nothing was added, weird... Let's increase the position to not remain stuck.
                parserState.pos += 1;
            }
            foundStopChar = false;
        }
        if (parserState.typeFilter !== null) {
            throw [
                "Unexpected ",
                ":",
                " (expected path after type filter ",
                parserState.typeFilter + ":",
                ")",
            ];
        }
        while (parserState.pos < parserState.length) {
            if (isReturnArrow(parserState)) {
                parserState.pos += 2;
                skipWhitespace(parserState);
                // Get returned elements.
                getItemsBefore(query, parserState, query.returned, "");
                // Nothing can come afterward!
                if (query.returned.length === 0) {
                    throw ["Expected at least one item after ", "->"];
                }
                break;
            } else {
                parserState.pos += 1;
            }
        }
    }

    /**
     * Takes the user search input and returns an empty `ParsedQuery`.
     *
     * @param {string} userQuery
     *
     * @return {ParsedQuery}
     */
    function newParsedQuery(userQuery) {
        return {
            original: userQuery,
            userQuery: userQuery.toLowerCase(),
            elems: [],
            returned: [],
            // Total number of "top" elements (does not include generics).
            foundElems: 0,
            // Total number of elements (includes generics).
            totalElems: 0,
            literalSearch: false,
            error: null,
            correction: null,
            proposeCorrectionFrom: null,
            proposeCorrectionTo: null,
        };
    }

    /**
     * Build an URL with search parameters.
     *
     * @param {string} search            - The current search being performed.
     * @param {string|null} filterCrates - The current filtering crate (if any).
     *
     * @return {string}
     */
    function buildUrl(search, filterCrates) {
        let extra = "?search=" + encodeURIComponent(search);

        if (filterCrates !== null) {
            extra += "&filter-crate=" + encodeURIComponent(filterCrates);
        }
        return getNakedUrl() + extra + window.location.hash;
    }

    /**
     * Return the filtering crate or `null` if there is none.
     *
     * @return {string|null}
     */
    function getFilterCrates() {
        const elem = document.getElementById("crate-search");

        if (elem &&
            elem.value !== "all crates" &&
            hasOwnPropertyRustdoc(rawSearchIndex, elem.value)
        ) {
            return elem.value;
        }
        return null;
    }

    /**
     * Parses the query.
     *
     * The supported syntax by this parser is given in the rustdoc book chapter
     * /src/doc/rustdoc/src/read-documentation/search.md
     *
     * When adding new things to the parser, add them there, too!
     *
     * @param  {string} val     - The user query
     *
     * @return {ParsedQuery}    - The parsed query
     */
    function parseQuery(userQuery) {
        function convertTypeFilterOnElem(elem) {
            if (elem.typeFilter !== null) {
                let typeFilter = elem.typeFilter;
                if (typeFilter === "const") {
                    typeFilter = "constant";
                }
                elem.typeFilter = itemTypeFromName(typeFilter);
            } else {
                elem.typeFilter = NO_TYPE_FILTER;
            }
            for (const elem2 of elem.generics) {
                convertTypeFilterOnElem(elem2);
            }
        }
        userQuery = userQuery.trim();
        const parserState = {
            length: userQuery.length,
            pos: 0,
            // Total number of elements (includes generics).
            totalElems: 0,
            genericsElems: 0,
            typeFilter: null,
            userQuery: userQuery.toLowerCase(),
        };
        let query = newParsedQuery(userQuery);

        try {
            parseInput(query, parserState);
            for (const elem of query.elems) {
                convertTypeFilterOnElem(elem);
            }
            for (const elem of query.returned) {
                convertTypeFilterOnElem(elem);
            }
        } catch (err) {
            query = newParsedQuery(userQuery);
            query.error = err;
            return query;
        }

        if (!query.literalSearch) {
            // If there is more than one element in the query, we switch to literalSearch in any
            // case.
            query.literalSearch = parserState.totalElems > 1;
        }
        query.foundElems = query.elems.length + query.returned.length;
        query.totalElems = parserState.totalElems;
        return query;
    }

    /**
     * Creates the query results.
     *
     * @param {Array<Result>} results_in_args
     * @param {Array<Result>} results_returned
     * @param {Array<Result>} results_others
     * @param {ParsedQuery} parsedQuery
     *
     * @return {ResultsTable}
     */
    function createQueryResults(results_in_args, results_returned, results_others, parsedQuery) {
        return {
            "in_args": results_in_args,
            "returned": results_returned,
            "others": results_others,
            "query": parsedQuery,
        };
    }

    /**
     * Executes the parsed query and builds a {ResultsTable}.
     *
     * @param  {ParsedQuery} parsedQuery - The parsed user query
     * @param  {Object} searchWords      - The list of search words to query against
     * @param  {Object} [filterCrates]   - Crate to search in if defined
     * @param  {Object} [currentCrate]   - Current crate, to rank results from this crate higher
     *
     * @return {ResultsTable}
     */
    function execQuery(parsedQuery, searchWords, filterCrates, currentCrate) {
        const results_others = new Map(), results_in_args = new Map(),
            results_returned = new Map();

        /**
         * Add extra data to result objects, and filter items that have been
         * marked for removal.
         *
         * @param {[ResultObject]} results
         * @returns {[ResultObject]}
         */
        function transformResults(results) {
            const duplicates = new Set();
            const out = [];

            for (const result of results) {
                if (result.id !== -1) {
                    const obj = searchIndex[result.id];
                    obj.dist = result.dist;
                    const res = buildHrefAndPath(obj);
                    obj.displayPath = pathSplitter(res[0]);
                    obj.fullPath = obj.displayPath + obj.name;
                    // To be sure than it some items aren't considered as duplicate.
                    obj.fullPath += "|" + obj.ty;

                    if (duplicates.has(obj.fullPath)) {
                        continue;
                    }
                    duplicates.add(obj.fullPath);

                    obj.href = res[1];
                    out.push(obj);
                    if (out.length >= MAX_RESULTS) {
                        break;
                    }
                }
            }
            return out;
        }

        /**
         * This function takes a result map, and sorts it by various criteria, including edit
         * distance, substring match, and the crate it comes from.
         *
         * @param {Results} results
         * @param {boolean} isType
         * @param {string} preferredCrate
         * @returns {[ResultObject]}
         */
        function sortResults(results, isType, preferredCrate) {
            // if there are no results then return to default and fail
            if (results.size === 0) {
                return [];
            }

            const userQuery = parsedQuery.userQuery;
            const result_list = [];
            for (const result of results.values()) {
                result.word = searchWords[result.id];
                result.item = searchIndex[result.id] || {};
                result_list.push(result);
            }

            result_list.sort((aaa, bbb) => {
                let a, b;

                // sort by exact match with regard to the last word (mismatch goes later)
                a = (aaa.word !== userQuery);
                b = (bbb.word !== userQuery);
                if (a !== b) {
                    return a - b;
                }

                // sort by index of keyword in item name (no literal occurrence goes later)
                a = (aaa.index < 0);
                b = (bbb.index < 0);
                if (a !== b) {
                    return a - b;
                }

                // Sort by distance in the path part, if specified
                // (less changes required to match means higher rankings)
                a = aaa.path_dist;
                b = bbb.path_dist;
                if (a !== b) {
                    return a - b;
                }

                // (later literal occurrence, if any, goes later)
                a = aaa.index;
                b = bbb.index;
                if (a !== b) {
                    return a - b;
                }

                // Sort by distance in the name part, the last part of the path
                // (less changes required to match means higher rankings)
                a = (aaa.dist);
                b = (bbb.dist);
                if (a !== b) {
                    return a - b;
                }

                // sort deprecated items later
                a = aaa.item.deprecated;
                b = bbb.item.deprecated;
                if (a !== b) {
                    return a - b;
                }

                // sort by crate (current crate comes first)
                a = (aaa.item.crate !== preferredCrate);
                b = (bbb.item.crate !== preferredCrate);
                if (a !== b) {
                    return a - b;
                }

                // sort by item name length (longer goes later)
                a = aaa.word.length;
                b = bbb.word.length;
                if (a !== b) {
                    return a - b;
                }

                // sort by item name (lexicographically larger goes later)
                a = aaa.word;
                b = bbb.word;
                if (a !== b) {
                    return (a > b ? +1 : -1);
                }

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
                if (a !== b) {
                    return a - b;
                }

                // sort by type (later occurrence in `itemTypes` goes later)
                a = aaa.item.ty;
                b = bbb.item.ty;
                if (a !== b) {
                    return a - b;
                }

                // sort by path (lexicographically larger goes later)
                a = aaa.item.path;
                b = bbb.item.path;
                if (a !== b) {
                    return (a > b ? +1 : -1);
                }

                // que sera, sera
                return 0;
            });

            let nameSplit = null;
            if (parsedQuery.elems.length === 1) {
                const hasPath = typeof parsedQuery.elems[0].path === "undefined";
                nameSplit = hasPath ? null : parsedQuery.elems[0].path;
            }

            for (const result of result_list) {
                // this validation does not make sense when searching by types
                if (result.dontValidate) {
                    continue;
                }
                const name = result.item.name.toLowerCase(),
                    path = result.item.path.toLowerCase(),
                    parent = result.item.parent;

                if (!isType && !validateResult(name, path, nameSplit, parent)) {
                    result.id = -1;
                }
            }
            return transformResults(result_list);
        }

        /**
         * This function checks if a list of search query `queryElems` can all be found in the
         * search index (`fnTypes`).
         *
         * This function returns `true` on a match, or `false` if none. If `solutionCb` is
         * supplied, it will call that function with mgens, and that callback can accept or
         * reject the result bu returning `true` or `false`. If the callback returns false,
         * then this function will try with a different solution, or bail with false if it
         * runs out of candidates.
         *
         * @param {Array<FunctionType>} fnTypes - The objects to check.
         * @param {Array<QueryElement>} queryElems - The elements from the parsed query.
         * @param {[FunctionType]} whereClause - Trait bounds for generic items.
         * @param {Map<number,number>|null} mgensIn
         *     - Map functions generics to query generics (never modified).
         * @param {null|Map<number,number> -> bool} solutionCb - Called for each `mgens` solution.
         *
         * @return {boolean} - Returns true if a match, false otherwise.
         */
        function unifyFunctionTypes(fnTypesIn, queryElems, whereClause, mgensIn, solutionCb) {
            /**
             * @type Map<integer, integer>
             */
            let mgens = new Map(mgensIn);
            if (queryElems.length === 0) {
                return !solutionCb || solutionCb(mgens);
            }
            if (!fnTypesIn || fnTypesIn.length === 0) {
                return false;
            }
            const ql = queryElems.length;
            let fl = fnTypesIn.length;
            /**
             * @type Array<FunctionType>
             */
            let fnTypes = fnTypesIn.slice();
            /**
             * loop works by building up a solution set in the working arrays
             * fnTypes gets mutated in place to make this work, while queryElems
             * is left alone
             *
             *                                  vvvvvvv `i` points here
             * queryElems = [ good, good, good, unknown, unknown ],
             * fnTypes    = [ good, good, good, unknown, unknown ],
             *                ----------------  ^^^^^^^^^^^^^^^^ `j` iterates after `i`,
             *                |                                   looking for candidates
             *                everything before `i` is the
             *                current working solution
             *
             * Everything in the current working solution is known to be a good
             * match, but it might not be the match we wind up going with, because
             * there might be more than one candidate match, and we need to try them all
             * before giving up. So, to handle this, it backtracks on failure.
             *
             * @type Array<{
             *     "fnTypesScratch": Array<FunctionType>,
             *     "queryElemsOffset": integer,
             *     "fnTypesOffset": integer
             * }>
             */
            const backtracking = [];
            let i = 0;
            let j = 0;
            const backtrack = () => {
                while (backtracking.length !== 0) {
                    // this session failed, but there are other possible solutions
                    // to backtrack, reset to (a copy of) the old array, do the swap or unboxing
                    const {
                        fnTypesScratch,
                        mgensScratch,
                        queryElemsOffset,
                        fnTypesOffset,
                        unbox,
                    } = backtracking.pop();
                    mgens = new Map(mgensScratch);
                    const fnType = fnTypesScratch[fnTypesOffset];
                    const queryElem = queryElems[queryElemsOffset];
                    if (unbox) {
                        if (fnType.id < 0) {
                            if (mgens.has(fnType.id) && mgens.get(fnType.id) !== 0) {
                                continue;
                            }
                            mgens.set(fnType.id, 0);
                        }
                        const generics = fnType.id < 0 ?
                            whereClause[(-fnType.id) - 1] :
                            fnType.generics;
                        fnTypes = fnTypesScratch.toSpliced(fnTypesOffset, 1, ...generics);
                        fl = fnTypes.length;
                        // re-run the matching algorithm on this item
                        i = queryElemsOffset - 1;
                    } else {
                        if (fnType.id < 0) {
                            if (mgens.has(fnType.id) && mgens.get(fnType.id) !== queryElem.id) {
                                continue;
                            }
                            mgens.set(fnType.id, queryElem.id);
                        }
                        fnTypes = fnTypesScratch.slice();
                        fl = fnTypes.length;
                        const tmp = fnTypes[queryElemsOffset];
                        fnTypes[queryElemsOffset] = fnTypes[fnTypesOffset];
                        fnTypes[fnTypesOffset] = tmp;
                        // this is known as a good match; go to the next one
                        i = queryElemsOffset;
                    }
                    return true;
                }
                return false;
            };
            for (i = 0; i !== ql; ++i) {
                const queryElem = queryElems[i];
                /**
                 * list of potential function types that go with the current query element.
                 * @type Array<integer>
                 */
                const matchCandidates = [];
                let fnTypesScratch = null;
                let mgensScratch = null;
                // don't try anything before `i`, because they've already been
                // paired off with the other query elements
                for (j = i; j !== fl; ++j) {
                    const fnType = fnTypes[j];
                    if (unifyFunctionTypeIsMatchCandidate(fnType, queryElem, whereClause, mgens)) {
                        if (!fnTypesScratch) {
                            fnTypesScratch = fnTypes.slice();
                        }
                        unifyFunctionTypes(
                            fnType.generics,
                            queryElem.generics,
                            whereClause,
                            mgens,
                            mgensScratch => {
                                matchCandidates.push({
                                    fnTypesScratch,
                                    mgensScratch,
                                    queryElemsOffset: i,
                                    fnTypesOffset: j,
                                    unbox: false,
                                });
                                return false; // "reject" all candidates to gather all of them
                            }
                        );
                    }
                    if (unifyFunctionTypeIsUnboxCandidate(fnType, queryElem, whereClause, mgens)) {
                        if (!fnTypesScratch) {
                            fnTypesScratch = fnTypes.slice();
                        }
                        if (!mgensScratch) {
                            mgensScratch = new Map(mgens);
                        }
                        backtracking.push({
                            fnTypesScratch,
                            mgensScratch,
                            queryElemsOffset: i,
                            fnTypesOffset: j,
                            unbox: true,
                        });
                    }
                }
                if (matchCandidates.length === 0) {
                    if (backtrack()) {
                        continue;
                    } else {
                        return false;
                    }
                }
                // use the current candidate
                const {fnTypesOffset: candidate, mgensScratch: mgensNew} = matchCandidates.pop();
                if (fnTypes[candidate].id < 0 && queryElems[i].id < 0) {
                    mgens.set(fnTypes[candidate].id, queryElems[i].id);
                }
                for (const [fid, qid] of mgensNew) {
                    mgens.set(fid, qid);
                }
                // `i` and `j` are paired off
                // `queryElems[i]` is left in place
                // `fnTypes[j]` is swapped with `fnTypes[i]` to pair them off
                const tmp = fnTypes[candidate];
                fnTypes[candidate] = fnTypes[i];
                fnTypes[i] = tmp;
                // write other candidates to backtracking queue
                for (const otherCandidate of matchCandidates) {
                    backtracking.push(otherCandidate);
                }
                // If we're on the last item, check the solution with the callback
                // backtrack if the callback says its unsuitable
                while (i === (ql - 1) && solutionCb && !solutionCb(mgens)) {
                    if (!backtrack()) {
                        return false;
                    }
                }
            }
            return true;
        }
        function unifyFunctionTypeIsMatchCandidate(fnType, queryElem, whereClause, mgens) {
            // type filters look like `trait:Read` or `enum:Result`
            if (!typePassesFilter(queryElem.typeFilter, fnType.ty)) {
                return false;
            }
            // fnType.id < 0 means generic
            // queryElem.id < 0 does too
            // mgens[fnType.id] = queryElem.id
            // or, if mgens[fnType.id] = 0, then we've matched this generic with a bare trait
            // and should make that same decision everywhere it appears
            if (fnType.id < 0 && queryElem.id < 0) {
                if (mgens.has(fnType.id) && mgens.get(fnType.id) !== queryElem.id) {
                    return false;
                }
                for (const [fid, qid] of mgens.entries()) {
                    if (fnType.id !== fid && queryElem.id === qid) {
                        return false;
                    }
                    if (fnType.id === fid && queryElem.id !== qid) {
                        return false;
                    }
                }
            } else {
                if (queryElem.id === typeNameIdOfArrayOrSlice &&
                    (fnType.id === typeNameIdOfSlice || fnType.id === typeNameIdOfArray)
                ) {
                    // [] matches primitive:array or primitive:slice
                    // if it matches, then we're fine, and this is an appropriate match candidate
                } else if (fnType.id !== queryElem.id || queryElem.id === null) {
                    return false;
                }
                // If the query elem has generics, and the function doesn't,
                // it can't match.
                if (fnType.generics.length === 0 && queryElem.generics.length !== 0) {
                    return false;
                }
                // If the query element is a path (it contains `::`), we need to check if this
                // path is compatible with the target type.
                const queryElemPathLength = queryElem.pathWithoutLast.length;
                if (queryElemPathLength > 0) {
                    const fnTypePath = fnType.path !== undefined && fnType.path !== null ?
                        fnType.path.split("::") : [];
                    // If the path provided in the query element is longer than this type,
                    // no need to check it since it won't match in any case.
                    if (queryElemPathLength > fnTypePath.length) {
                        return false;
                    }
                    let i = 0;
                    for (const path of fnTypePath) {
                        if (path === queryElem.pathWithoutLast[i]) {
                            i += 1;
                            if (i >= queryElemPathLength) {
                                break;
                            }
                        }
                    }
                    if (i < queryElemPathLength) {
                        // If we didn't find all parts of the path of the query element inside
                        // the fn type, then it's not the right one.
                        return false;
                    }
                }
            }
            return true;
        }
        function unifyFunctionTypeIsUnboxCandidate(fnType, queryElem, whereClause, mgens) {
            if (fnType.id < 0 && queryElem.id >= 0) {
                if (!whereClause) {
                    return false;
                }
                // mgens[fnType.id] === 0 indicates that we committed to unboxing this generic
                // mgens[fnType.id] === null indicates that we haven't decided yet
                if (mgens.has(fnType.id) && mgens.get(fnType.id) !== 0) {
                    return false;
                }
                // This is only a potential unbox if the search query appears in the where clause
                // for example, searching `Read -> usize` should find
                // `fn read_all<R: Read>(R) -> Result<usize>`
                // generic `R` is considered "unboxed"
                return checkIfInList(whereClause[(-fnType.id) - 1], queryElem, whereClause);
            } else if (fnType.generics && fnType.generics.length > 0) {
                return checkIfInList(fnType.generics, queryElem, whereClause);
            }
            return false;
        }

        /**
          * This function checks if the object (`row`) matches the given type (`elem`) and its
          * generics (if any).
          *
          * @param {Array<FunctionType>} list
          * @param {QueryElement} elem          - The element from the parsed query.
          * @param {[FunctionType]} whereClause - Trait bounds for generic items.
          *
          * @return {boolean} - Returns true if found, false otherwise.
          */
        function checkIfInList(list, elem, whereClause) {
            for (const entry of list) {
                if (checkType(entry, elem, whereClause)) {
                    return true;
                }
            }
            return false;
        }

        /**
          * This function checks if the object (`row`) matches the given type (`elem`) and its
          * generics (if any).
          *
          * @param {Row} row
          * @param {QueryElement} elem          - The element from the parsed query.
          * @param {[FunctionType]} whereClause - Trait bounds for generic items.
          *
          * @return {boolean} - Returns true if the type matches, false otherwise.
          */
        function checkType(row, elem, whereClause) {
            if (elem.id < 0) {
                return row.id < 0 || checkIfInList(row.generics, elem, whereClause);
            }
            if (row.id > 0 && elem.id > 0 && elem.pathWithoutLast.length === 0 &&
                typePassesFilter(elem.typeFilter, row.ty) && elem.generics.length === 0 &&
                // special case
                elem.id !== typeNameIdOfArrayOrSlice
            ) {
                return row.id === elem.id || checkIfInList(row.generics, elem, whereClause);
            }
            return unifyFunctionTypes([row], [elem], whereClause);
        }

        function checkPath(contains, ty, maxEditDistance) {
            if (contains.length === 0) {
                return 0;
            }
            let ret_dist = maxEditDistance + 1;
            const path = ty.path.split("::");

            if (ty.parent && ty.parent.name) {
                path.push(ty.parent.name.toLowerCase());
            }

            const length = path.length;
            const clength = contains.length;
            if (clength > length) {
                return maxEditDistance + 1;
            }
            for (let i = 0; i < length; ++i) {
                if (i + clength > length) {
                    break;
                }
                let dist_total = 0;
                let aborted = false;
                for (let x = 0; x < clength; ++x) {
                    const dist = editDistance(path[i + x], contains[x], maxEditDistance);
                    if (dist > maxEditDistance) {
                        aborted = true;
                        break;
                    }
                    dist_total += dist;
                }
                if (!aborted) {
                    ret_dist = Math.min(ret_dist, Math.round(dist_total / clength));
                }
            }
            return ret_dist;
        }

        function typePassesFilter(filter, type) {
            // No filter or Exact mach
            if (filter <= NO_TYPE_FILTER || filter === type) return true;

            // Match related items
            const name = itemTypes[type];
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
                deprecated: item.deprecated,
                implDisambiguator: item.implDisambiguator,
            };
        }

        function handleAliases(ret, query, filterCrates, currentCrate) {
            const lowerQuery = query.toLowerCase();
            // We separate aliases and crate aliases because we want to have current crate
            // aliases to be before the others in the displayed results.
            const aliases = [];
            const crateAliases = [];
            if (filterCrates !== null) {
                if (ALIASES.has(filterCrates) && ALIASES.get(filterCrates).has(lowerQuery)) {
                    const query_aliases = ALIASES.get(filterCrates).get(lowerQuery);
                    for (const alias of query_aliases) {
                        aliases.push(createAliasFromItem(searchIndex[alias]));
                    }
                }
            } else {
                for (const [crate, crateAliasesIndex] of ALIASES) {
                    if (crateAliasesIndex.has(lowerQuery)) {
                        const pushTo = crate === currentCrate ? crateAliases : aliases;
                        const query_aliases = crateAliasesIndex.get(lowerQuery);
                        for (const alias of query_aliases) {
                            pushTo.push(createAliasFromItem(searchIndex[alias]));
                        }
                    }
                }
            }

            const sortFunc = (aaa, bbb) => {
                if (aaa.path < bbb.path) {
                    return 1;
                } else if (aaa.path === bbb.path) {
                    return 0;
                }
                return -1;
            };
            crateAliases.sort(sortFunc);
            aliases.sort(sortFunc);

            const pushFunc = alias => {
                alias.alias = query;
                const res = buildHrefAndPath(alias);
                alias.displayPath = pathSplitter(res[0]);
                alias.fullPath = alias.displayPath + alias.name;
                alias.href = res[1];

                ret.others.unshift(alias);
                if (ret.others.length > MAX_RESULTS) {
                    ret.others.pop();
                }
            };

            aliases.forEach(pushFunc);
            crateAliases.forEach(pushFunc);
        }

        /**
         * This function adds the given result into the provided `results` map if it matches the
         * following condition:
         *
         * * If it is a "literal search" (`parsedQuery.literalSearch`), then `dist` must be 0.
         * * If it is not a "literal search", `dist` must be <= `maxEditDistance`.
         *
         * The `results` map contains information which will be used to sort the search results:
         *
         * * `fullId` is a `string`` used as the key of the object we use for the `results` map.
         * * `id` is the index in both `searchWords` and `searchIndex` arrays for this element.
         * * `index` is an `integer`` used to sort by the position of the word in the item's name.
         * * `dist` is the main metric used to sort the search results.
         * * `path_dist` is zero if a single-component search query is used, otherwise it's the
         *   distance computed for everything other than the last path component.
         *
         * @param {Results} results
         * @param {string} fullId
         * @param {integer} id
         * @param {integer} index
         * @param {integer} dist
         * @param {integer} path_dist
         */
        function addIntoResults(results, fullId, id, index, dist, path_dist, maxEditDistance) {
            const inBounds = dist <= maxEditDistance || index !== -1;
            if (dist === 0 || (!parsedQuery.literalSearch && inBounds)) {
                if (results.has(fullId)) {
                    const result = results.get(fullId);
                    if (result.dontValidate || result.dist <= dist) {
                        return;
                    }
                }
                results.set(fullId, {
                    id: id,
                    index: index,
                    dontValidate: parsedQuery.literalSearch,
                    dist: dist,
                    path_dist: path_dist,
                });
            }
        }

        /**
         * This function is called in case the query is only one element (with or without generics).
         * This element will be compared to arguments' and returned values' items and also to items.
         *
         * Other important thing to note: since there is only one element, we use edit
         * distance for name comparisons.
         *
         * @param {Row} row
         * @param {integer} pos              - Position in the `searchIndex`.
         * @param {QueryElement} elem        - The element from the parsed query.
         * @param {Results} results_others   - Unqualified results (not in arguments nor in
         *                                     returned values).
         * @param {Results} results_in_args  - Matching arguments results.
         * @param {Results} results_returned - Matching returned arguments results.
         */
        function handleSingleArg(
            row,
            pos,
            elem,
            results_others,
            results_in_args,
            results_returned,
            maxEditDistance
        ) {
            if (!row || (filterCrates !== null && row.crate !== filterCrates)) {
                return;
            }
            let index = -1, path_dist = 0;
            const fullId = row.id;
            const searchWord = searchWords[pos];

            const in_args = row.type && row.type.inputs
                && checkIfInList(row.type.inputs, elem, row.type.where_clause);
            if (in_args) {
                // path_dist is 0 because no parent path information is currently stored
                // in the search index
                addIntoResults(results_in_args, fullId, pos, -1, 0, 0, maxEditDistance);
            }
            const returned = row.type && row.type.output
                && checkIfInList(row.type.output, elem, row.type.where_clause);
            if (returned) {
                addIntoResults(results_returned, fullId, pos, -1, 0, 0, maxEditDistance);
            }

            if (!typePassesFilter(elem.typeFilter, row.ty)) {
                return;
            }

            const row_index = row.normalizedName.indexOf(elem.pathLast);
            const word_index = searchWord.indexOf(elem.pathLast);

            // lower indexes are "better" matches
            // rank based on the "best" match
            if (row_index === -1) {
                index = word_index;
            } else if (word_index === -1) {
                index = row_index;
            } else if (word_index < row_index) {
                index = word_index;
            } else {
                index = row_index;
            }

            if (elem.fullPath.length > 1) {
                path_dist = checkPath(elem.pathWithoutLast, row, maxEditDistance);
                if (path_dist > maxEditDistance) {
                    return;
                }
            }

            if (parsedQuery.literalSearch) {
                if (searchWord === elem.name) {
                    addIntoResults(results_others, fullId, pos, index, 0, path_dist);
                }
                return;
            }

            const dist = editDistance(searchWord, elem.pathLast, maxEditDistance);

            if (index === -1 && dist + path_dist > maxEditDistance) {
                return;
            }

            addIntoResults(results_others, fullId, pos, index, dist, path_dist, maxEditDistance);
        }

        /**
         * This function is called in case the query has more than one element. In this case, it'll
         * try to match the items which validates all the elements. For `aa -> bb` will look for
         * functions which have a parameter `aa` and has `bb` in its returned values.
         *
         * @param {Row} row
         * @param {integer} pos      - Position in the `searchIndex`.
         * @param {Object} results
         */
        function handleArgs(row, pos, results) {
            if (!row || (filterCrates !== null && row.crate !== filterCrates) || !row.type) {
                return;
            }

            // If the result is too "bad", we return false and it ends this search.
            if (!unifyFunctionTypes(
                row.type.inputs,
                parsedQuery.elems,
                row.type.where_clause,
                null,
                mgens => {
                    return unifyFunctionTypes(
                        row.type.output,
                        parsedQuery.returned,
                        row.type.where_clause,
                        mgens
                    );
                }
            )) {
                return;
            }

            addIntoResults(results, row.id, pos, 0, 0, 0, Number.MAX_VALUE);
        }

        function innerRunQuery() {
            let elem, i, nSearchWords, in_returned, row;

            let queryLen = 0;
            for (const elem of parsedQuery.elems) {
                queryLen += elem.name.length;
            }
            for (const elem of parsedQuery.returned) {
                queryLen += elem.name.length;
            }
            const maxEditDistance = Math.floor(queryLen / 3);

            /**
             * @type {Map<string, integer>}
             */
            const genericSymbols = new Map();

            /**
             * Convert names to ids in parsed query elements.
             * This is not used for the "In Names" tab, but is used for the
             * "In Params", "In Returns", and "In Function Signature" tabs.
             *
             * If there is no matching item, but a close-enough match, this
             * function also that correction.
             *
             * See `buildTypeMapIndex` for more information.
             *
             * @param {QueryElement} elem
             */
            function convertNameToId(elem) {
                if (typeNameIdMap.has(elem.pathLast)) {
                    elem.id = typeNameIdMap.get(elem.pathLast);
                } else if (!parsedQuery.literalSearch) {
                    let match = null;
                    let matchDist = maxEditDistance + 1;
                    let matchName = "";
                    for (const [name, id] of typeNameIdMap) {
                        const dist = editDistance(name, elem.pathLast, maxEditDistance);
                        if (dist <= matchDist && dist <= maxEditDistance) {
                            if (dist === matchDist && matchName > name) {
                                continue;
                            }
                            match = id;
                            matchDist = dist;
                            matchName = name;
                        }
                    }
                    if (match !== null) {
                        parsedQuery.correction = matchName;
                    }
                    elem.id = match;
                }
                if ((elem.id === null && parsedQuery.totalElems > 1 && elem.typeFilter === -1
                     && elem.generics.length === 0)
                    || elem.typeFilter === TY_GENERIC) {
                    if (genericSymbols.has(elem.name)) {
                        elem.id = genericSymbols.get(elem.name);
                    } else {
                        elem.id = -(genericSymbols.size + 1);
                        genericSymbols.set(elem.name, elem.id);
                    }
                    if (elem.typeFilter === -1 && elem.name.length >= 3) {
                        // Silly heuristic to catch if the user probably meant
                        // to not write a generic parameter. We don't use it,
                        // just bring it up.
                        const maxPartDistance = Math.floor(elem.name.length / 3);
                        let matchDist = maxPartDistance + 1;
                        let matchName = "";
                        for (const name of typeNameIdMap.keys()) {
                            const dist = editDistance(name, elem.name, maxPartDistance);
                            if (dist <= matchDist && dist <= maxPartDistance) {
                                if (dist === matchDist && matchName > name) {
                                    continue;
                                }
                                matchDist = dist;
                                matchName = name;
                            }
                        }
                        if (matchName !== "") {
                            parsedQuery.proposeCorrectionFrom = elem.name;
                            parsedQuery.proposeCorrectionTo = matchName;
                        }
                    }
                    elem.typeFilter = TY_GENERIC;
                }
                if (elem.generics.length > 0 && elem.typeFilter === TY_GENERIC) {
                    // Rust does not have HKT
                    parsedQuery.error = [
                        "Generic type parameter ",
                        elem.name,
                        " does not accept generic parameters",
                    ];
                }
                for (const elem2 of elem.generics) {
                    convertNameToId(elem2);
                }
            }

            for (const elem of parsedQuery.elems) {
                convertNameToId(elem);
            }
            for (const elem of parsedQuery.returned) {
                convertNameToId(elem);
            }

            if (parsedQuery.foundElems === 1) {
                if (parsedQuery.elems.length === 1) {
                    elem = parsedQuery.elems[0];
                    for (i = 0, nSearchWords = searchWords.length; i < nSearchWords; ++i) {
                        // It means we want to check for this element everywhere (in names, args and
                        // returned).
                        handleSingleArg(
                            searchIndex[i],
                            i,
                            elem,
                            results_others,
                            results_in_args,
                            results_returned,
                            maxEditDistance
                        );
                    }
                } else if (parsedQuery.returned.length === 1) {
                    // We received one returned argument to check, so looking into returned values.
                    elem = parsedQuery.returned[0];
                    for (i = 0, nSearchWords = searchWords.length; i < nSearchWords; ++i) {
                        row = searchIndex[i];
                        in_returned = row.type && unifyFunctionTypes(
                            row.type.output,
                            parsedQuery.returned,
                            row.type.where_clause
                        );
                        if (in_returned) {
                            addIntoResults(
                                results_others,
                                row.id,
                                i,
                                -1,
                                0,
                                Number.MAX_VALUE
                            );
                        }
                    }
                }
            } else if (parsedQuery.foundElems > 0) {
                for (i = 0, nSearchWords = searchWords.length; i < nSearchWords; ++i) {
                    handleArgs(searchIndex[i], i, results_others);
                }
            }
        }

        if (parsedQuery.error === null) {
            innerRunQuery();
        }

        const ret = createQueryResults(
            sortResults(results_in_args, true, currentCrate),
            sortResults(results_returned, true, currentCrate),
            sortResults(results_others, false, currentCrate),
            parsedQuery);
        handleAliases(ret, parsedQuery.original.replace(/"/g, ""), filterCrates, currentCrate);
        if (parsedQuery.error !== null && ret.others.length !== 0) {
            // It means some doc aliases were found so let's "remove" the error!
            ret.query.error = null;
        }
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
     * @param  {string} name   - The name of the result
     * @param  {string} path   - The path of the result
     * @param  {string} keys   - The keys to be used (["file", "open"])
     * @param  {Object} parent - The parent of the result
     *
     * @return {boolean}       - Whether the result is valid or not
     */
    function validateResult(name, path, keys, parent, maxEditDistance) {
        if (!keys || !keys.length) {
            return true;
        }
        for (const key of keys) {
            // each check is for validation so we negate the conditions and invalidate
            if (!(
                // check for an exact name match
                name.indexOf(key) > -1 ||
                // then an exact path match
                path.indexOf(key) > -1 ||
                // next if there is a parent, check for exact parent match
                (parent !== undefined && parent.name !== undefined &&
                    parent.name.toLowerCase().indexOf(key) > -1) ||
                // lastly check to see if the name was an editDistance match
                editDistance(name, key, maxEditDistance) <= maxEditDistance)) {
                return false;
            }
        }
        return true;
    }

    function nextTab(direction) {
        const next = (searchState.currentTab + direction + 3) % searchState.focusedByTab.length;
        searchState.focusedByTab[searchState.currentTab] = document.activeElement;
        printTab(next);
        focusSearchResult();
    }

    // Focus the first search result on the active tab, or the result that
    // was focused last time this tab was active.
    function focusSearchResult() {
        const target = searchState.focusedByTab[searchState.currentTab] ||
            document.querySelectorAll(".search-results.active a").item(0) ||
            document.querySelectorAll("#search-tabs button").item(searchState.currentTab);
        searchState.focusedByTab[searchState.currentTab] = null;
        if (target) {
            target.focus();
        }
    }

    function buildHrefAndPath(item) {
        let displayPath;
        let href;
        const type = itemTypes[item.ty];
        const name = item.name;
        let path = item.path;

        if (type === "mod") {
            displayPath = path + "::";
            href = ROOT_PATH + path.replace(/::/g, "/") + "/" +
                name + "/index.html";
        } else if (type === "import") {
            displayPath = item.path + "::";
            href = ROOT_PATH + item.path.replace(/::/g, "/") + "/index.html#reexport." + name;
        } else if (type === "primitive" || type === "keyword") {
            displayPath = "";
            href = ROOT_PATH + path.replace(/::/g, "/") +
                "/" + type + "." + name + ".html";
        } else if (type === "externcrate") {
            displayPath = "";
            href = ROOT_PATH + name + "/index.html";
        } else if (item.parent !== undefined) {
            const myparent = item.parent;
            let anchor = type + "." + name;
            const parentType = itemTypes[myparent.ty];
            let pageType = parentType;
            let pageName = myparent.name;

            if (parentType === "primitive") {
                displayPath = myparent.name + "::";
            } else if (type === "structfield" && parentType === "variant") {
                // Structfields belonging to variants are special: the
                // final path element is the enum name.
                const enumNameIdx = item.path.lastIndexOf("::");
                const enumName = item.path.substr(enumNameIdx + 2);
                path = item.path.substr(0, enumNameIdx);
                displayPath = path + "::" + enumName + "::" + myparent.name + "::";
                anchor = "variant." + myparent.name + ".field." + name;
                pageType = "enum";
                pageName = enumName;
            } else {
                displayPath = path + "::" + myparent.name + "::";
            }
            if (item.implDisambiguator !== null) {
                anchor = item.implDisambiguator + "/" + anchor;
            }
            href = ROOT_PATH + path.replace(/::/g, "/") +
                "/" + pageType +
                "." + pageName +
                ".html#" + anchor;
        } else {
            displayPath = item.path + "::";
            href = ROOT_PATH + item.path.replace(/::/g, "/") +
                "/" + type + "." + name + ".html";
        }
        return [displayPath, href];
    }

    function pathSplitter(path) {
        const tmp = "<span>" + path.replace(/::/g, "::</span><span>");
        if (tmp.endsWith("<span>")) {
            return tmp.slice(0, tmp.length - 6);
        }
        return tmp;
    }

    /**
     * Render a set of search results for a single tab.
     * @param {Array<?>}    array   - The search results for this tab
     * @param {ParsedQuery} query
     * @param {boolean}     display - True if this is the active tab
     */
    function addTab(array, query, display) {
        let extraClass = "";
        if (display === true) {
            extraClass = " active";
        }

        const output = document.createElement("div");
        let length = 0;
        if (array.length > 0) {
            output.className = "search-results " + extraClass;

            array.forEach(item => {
                const name = item.name;
                const type = itemTypes[item.ty];
                const longType = longItemTypes[item.ty];
                const typeName = longType.length !== 0 ? `${longType}` : "?";

                length += 1;

                const link = document.createElement("a");
                link.className = "result-" + type;
                link.href = item.href;

                const resultName = document.createElement("div");
                resultName.className = "result-name";

                resultName.insertAdjacentHTML(
                    "beforeend",
                    `<span class="typename">${typeName}</span>`);
                link.appendChild(resultName);

                let alias = " ";
                if (item.is_alias) {
                    alias = ` <div class="alias">\
<b>${item.alias}</b><i class="grey">&nbsp;- see&nbsp;</i>\
</div>`;
                }
                resultName.insertAdjacentHTML(
                    "beforeend",
                    `<div class="path">${alias}\
${item.displayPath}<span class="${type}">${name}</span>\
</div>`);

                const description = document.createElement("div");
                description.className = "desc";
                description.insertAdjacentHTML("beforeend", item.desc);

                link.appendChild(description);
                output.appendChild(link);
            });
        } else if (query.error === null) {
            output.className = "search-failed" + extraClass;
            output.innerHTML = "No results :(<br/>" +
                "Try on <a href=\"https://duckduckgo.com/?q=" +
                encodeURIComponent("rust " + query.userQuery) +
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
        // https://blog.horizon-eda.org/misc/2020/02/19/ui.html
        //
        // CSS runs with `font-variant-numeric: tabular-nums` to ensure all
        // digits are the same width. \u{2007} is a Unicode space character
        // that is defined to be the same width as a digit.
        const fmtNbElems =
            nbElems < 10  ? `\u{2007}(${nbElems})\u{2007}\u{2007}` :
            nbElems < 100 ? `\u{2007}(${nbElems})\u{2007}` :
            `\u{2007}(${nbElems})`;
        if (searchState.currentTab === tabNb) {
            return "<button class=\"selected\">" + text +
                   "<span class=\"count\">" + fmtNbElems + "</span></button>";
        }
        return "<button>" + text + "<span class=\"count\">" + fmtNbElems + "</span></button>";
    }

    /**
     * @param {ResultsTable} results
     * @param {boolean} go_to_first
     * @param {string} filterCrates
     */
    function showResults(results, go_to_first, filterCrates) {
        const search = searchState.outputElement();
        if (go_to_first || (results.others.length === 1
            && getSettingValue("go-to-only-result") === "true")
        ) {
            // Needed to force re-execution of JS when coming back to a page. Let's take this
            // scenario as example:
            //
            // 1. You have the "Directly go to item in search if there is only one result" option
            //    enabled.
            // 2. You make a search which results only one result, leading you automatically to
            //    this result.
            // 3. You go back to previous page.
            //
            // Now, without the call below, the JS will not be re-executed and the previous state
            // will be used, starting search again since the search input is not empty, leading you
            // back to the previous page again.
            window.onunload = () => {};
            searchState.removeQueryParameters();
            const elem = document.createElement("a");
            elem.href = results.others[0].href;
            removeClass(elem, "active");
            // For firefox, we need the element to be in the DOM so it can be clicked.
            document.body.appendChild(elem);
            elem.click();
            return;
        }
        if (results.query === undefined) {
            results.query = parseQuery(searchState.input.value);
        }

        currentResults = results.query.userQuery;

        const ret_others = addTab(results.others, results.query, true);
        const ret_in_args = addTab(results.in_args, results.query, false);
        const ret_returned = addTab(results.returned, results.query, false);

        // Navigate to the relevant tab if the current tab is empty, like in case users search
        // for "-> String". If they had selected another tab previously, they have to click on
        // it again.
        let currentTab = searchState.currentTab;
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

        let crates = "";
        const crates_list = Object.keys(rawSearchIndex);
        if (crates_list.length > 1) {
            crates = " in&nbsp;<div id=\"crate-search-div\"><select id=\"crate-search\">" +
                "<option value=\"all crates\">all crates</option>";
            for (const c of crates_list) {
                crates += `<option value="${c}" ${c === filterCrates && "selected"}>${c}</option>`;
            }
            crates += "</select></div>";
        }

        let output = `<h1 class="search-results-title">Results${crates}</h1>`;
        if (results.query.error !== null) {
            const error = results.query.error;
            error.forEach((value, index) => {
                value = value.split("<").join("&lt;").split(">").join("&gt;");
                if (index % 2 !== 0) {
                    error[index] = `<code>${value.replaceAll(" ", "&nbsp;")}</code>`;
                } else {
                    error[index] = value;
                }
            });
            output += `<h3 class="error">Query parser error: "${error.join("")}".</h3>`;
            output += "<div id=\"search-tabs\">" +
                makeTabHeader(0, "In Names", ret_others[1]) +
                "</div>";
            currentTab = 0;
        } else if (results.query.foundElems <= 1 && results.query.returned.length === 0) {
            output += "<div id=\"search-tabs\">" +
                makeTabHeader(0, "In Names", ret_others[1]) +
                makeTabHeader(1, "In Parameters", ret_in_args[1]) +
                makeTabHeader(2, "In Return Types", ret_returned[1]) +
                "</div>";
        } else {
            const signatureTabTitle =
                results.query.elems.length === 0 ? "In Function Return Types" :
                results.query.returned.length === 0 ? "In Function Parameters" :
                "In Function Signatures";
            output += "<div id=\"search-tabs\">" +
                makeTabHeader(0, signatureTabTitle, ret_others[1]) +
                "</div>";
            currentTab = 0;
        }

        if (results.query.correction !== null) {
            const orig = results.query.returned.length > 0
                ? results.query.returned[0].name
                : results.query.elems[0].name;
            output += "<h3 class=\"search-corrections\">" +
                `Type "${orig}" not found. ` +
                "Showing results for closest type name " +
                `"${results.query.correction}" instead.</h3>`;
        }
        if (results.query.proposeCorrectionFrom !== null) {
            const orig = results.query.proposeCorrectionFrom;
            const targ = results.query.proposeCorrectionTo;
            output += "<h3 class=\"search-corrections\">" +
                `Type "${orig}" not found and used as generic parameter. ` +
                `Consider searching for "${targ}" instead.</h3>`;
        }

        const resultsElem = document.createElement("div");
        resultsElem.id = "results";
        resultsElem.appendChild(ret_others[0]);
        resultsElem.appendChild(ret_in_args[0]);
        resultsElem.appendChild(ret_returned[0]);

        search.innerHTML = output;
        const crateSearch = document.getElementById("crate-search");
        if (crateSearch) {
            crateSearch.addEventListener("input", updateCrate);
        }
        search.appendChild(resultsElem);
        // Reset focused elements.
        searchState.showResults(search);
        const elems = document.getElementById("search-tabs").childNodes;
        searchState.focusedByTab = [];
        let i = 0;
        for (const elem of elems) {
            const j = i;
            elem.onclick = () => printTab(j);
            searchState.focusedByTab.push(null);
            i += 1;
        }
        printTab(currentTab);
    }

    function updateSearchHistory(url) {
        if (!browserSupportsHistoryApi()) {
            return;
        }
        const params = searchState.getQueryStringParams();
        if (!history.state && !params.search) {
            history.pushState(null, "", url);
        } else {
            history.replaceState(null, "", url);
        }
    }

    /**
     * Perform a search based on the current state of the search input element
     * and display the results.
     * @param {Event}   [e]       - The event that triggered this search, if any
     * @param {boolean} [forced]
     */
    function search(e, forced) {
        if (e) {
            e.preventDefault();
        }
        const query = parseQuery(searchState.input.value.trim());
        let filterCrates = getFilterCrates();

        if (!forced && query.userQuery === currentResults) {
            if (query.userQuery.length > 0) {
                putBackSearch();
            }
            return;
        }

        searchState.setLoadingSearch();

        const params = searchState.getQueryStringParams();

        // In case we have no information about the saved crate and there is a URL query parameter,
        // we override it with the URL query parameter.
        if (filterCrates === null && params["filter-crate"] !== undefined) {
            filterCrates = params["filter-crate"];
        }

        // Update document title to maintain a meaningful browser history
        searchState.title = "Results for " + query.original + " - Rust";

        // Because searching is incremental by character, only the most
        // recent search query is added to the browser history.
        updateSearchHistory(buildUrl(query.original, filterCrates));

        showResults(
            execQuery(query, searchWords, filterCrates, window.currentCrate),
            params.go_to_first,
            filterCrates);
    }

    /**
     * Convert a list of RawFunctionType / ID to object-based FunctionType.
     *
     * Crates often have lots of functions in them, and it's common to have a large number of
     * functions that operate on a small set of data types, so the search index compresses them
     * by encoding function parameter and return types as indexes into an array of names.
     *
     * Even when a general-purpose compression algorithm is used, this is still a win. I checked.
     * https://github.com/rust-lang/rust/pull/98475#issue-1284395985
     *
     * The format for individual function types is encoded in
     * librustdoc/html/render/mod.rs: impl Serialize for RenderType
     *
     * @param {null|Array<RawFunctionType>} types
     * @param {Array<{name: string, ty: number}>} lowercasePaths
     *
     * @return {Array<FunctionSearchType>}
     */
    function buildItemSearchTypeAll(types, lowercasePaths) {
        return types.map(type => buildItemSearchType(type, lowercasePaths));
    }

    /**
     * Converts a single type.
     *
     * @param {RawFunctionType} type
     */
    function buildItemSearchType(type, lowercasePaths) {
        const PATH_INDEX_DATA = 0;
        const GENERICS_DATA = 1;
        let pathIndex, generics;
        if (typeof type === "number") {
            pathIndex = type;
            generics = [];
        } else {
            pathIndex = type[PATH_INDEX_DATA];
            generics = buildItemSearchTypeAll(
                type[GENERICS_DATA],
                lowercasePaths
            );
        }
        if (pathIndex < 0) {
            // types less than 0 are generic parameters
            // the actual names of generic parameters aren't stored, since they aren't API
            return {
                id: pathIndex,
                ty: TY_GENERIC,
                path: null,
                generics,
            };
        }
        if (pathIndex === 0) {
            // `0` is used as a sentinel because it's fewer bytes than `null`
            return {
                id: null,
                ty: null,
                path: null,
                generics,
            };
        }
        const item = lowercasePaths[pathIndex - 1];
        return {
            id: buildTypeMapIndex(item.name),
            ty: item.ty,
            path: item.path,
            generics,
        };
    }

    /**
     * Convert from RawFunctionSearchType to FunctionSearchType.
     *
     * Crates often have lots of functions in them, and function signatures are sometimes complex,
     * so rustdoc uses a pretty tight encoding for them. This function converts it to a simpler,
     * object-based encoding so that the actual search code is more readable and easier to debug.
     *
     * The raw function search type format is generated using serde in
     * librustdoc/html/render/mod.rs: impl Serialize for IndexItemFunctionType
     *
     * @param {RawFunctionSearchType} functionSearchType
     * @param {Array<{name: string, ty: number}>} lowercasePaths
     * @param {Map<string, integer>}
     *
     * @return {null|FunctionSearchType}
     */
    function buildFunctionSearchType(functionSearchType, lowercasePaths) {
        const INPUTS_DATA = 0;
        const OUTPUT_DATA = 1;
        // `0` is used as a sentinel because it's fewer bytes than `null`
        if (functionSearchType === 0) {
            return null;
        }
        let inputs, output;
        if (typeof functionSearchType[INPUTS_DATA] === "number") {
            inputs = [buildItemSearchType(functionSearchType[INPUTS_DATA], lowercasePaths)];
        } else {
            inputs = buildItemSearchTypeAll(
                functionSearchType[INPUTS_DATA],
                lowercasePaths
            );
        }
        if (functionSearchType.length > 1) {
            if (typeof functionSearchType[OUTPUT_DATA] === "number") {
                output = [buildItemSearchType(functionSearchType[OUTPUT_DATA], lowercasePaths)];
            } else {
                output = buildItemSearchTypeAll(
                    functionSearchType[OUTPUT_DATA],
                    lowercasePaths
                );
            }
        } else {
            output = [];
        }
        const where_clause = [];
        const l = functionSearchType.length;
        for (let i = 2; i < l; ++i) {
            where_clause.push(typeof functionSearchType[i] === "number"
                ? [buildItemSearchType(functionSearchType[i], lowercasePaths)]
                : buildItemSearchTypeAll(functionSearchType[i], lowercasePaths));
        }
        return {
            inputs, output, where_clause,
        };
    }

    function buildIndex(rawSearchIndex) {
        searchIndex = [];
        /**
         * List of normalized search words (ASCII lowercased, and undescores removed).
         *
         * @type {Array<string>}
         */
        const searchWords = [];
        typeNameIdMap = new Map();
        const charA = "A".charCodeAt(0);
        let currentIndex = 0;
        let id = 0;

        // Initialize type map indexes for primitive list types
        // that can be searched using `[]` syntax.
        typeNameIdOfArray = buildTypeMapIndex("array");
        typeNameIdOfSlice = buildTypeMapIndex("slice");
        typeNameIdOfArrayOrSlice = buildTypeMapIndex("[]");

        for (const crate in rawSearchIndex) {
            if (!hasOwnPropertyRustdoc(rawSearchIndex, crate)) {
                continue;
            }

            let crateSize = 0;

            /**
             * The raw search data for a given crate. `n`, `t`, `d`, `i`, and `f`
             * are arrays with the same length. `q`, `a`, and `c` use a sparse
             * representation for compactness.
             *
             * `n[i]` contains the name of an item.
             *
             * `t[i]` contains the type of that item
             * (as a string of characters that represent an offset in `itemTypes`).
             *
             * `d[i]` contains the description of that item.
             *
             * `q` contains the full paths of the items. For compactness, it is a set of
             * (index, path) pairs used to create a map. If a given index `i` is
             * not present, this indicates "same as the last index present".
             *
             * `i[i]` contains an item's parent, usually a module. For compactness,
             * it is a set of indexes into the `p` array.
             *
             * `f[i]` contains function signatures, or `0` if the item isn't a function.
             * Functions are themselves encoded as arrays. The first item is a list of
             * types representing the function's inputs, and the second list item is a list
             * of types representing the function's output. Tuples are flattened.
             * Types are also represented as arrays; the first item is an index into the `p`
             * array, while the second is a list of types representing any generic parameters.
             *
             * b[i] contains an item's impl disambiguator. This is only present if an item
             * is defined in an impl block and, the impl block's type has more than one associated
             * item with the same name.
             *
             * `a` defines aliases with an Array of pairs: [name, offset], where `offset`
             * points into the n/t/d/q/i/f arrays.
             *
             * `doc` contains the description of the crate.
             *
             * `p` is a list of path/type pairs. It is used for parents and function parameters.
             *
             * `c` is an array of item indices that are deprecated.
             *
             * @type {{
             *   doc: string,
             *   a: Object,
             *   n: Array<string>,
             *   t: String,
             *   d: Array<string>,
             *   q: Array<[Number, string]>,
             *   i: Array<Number>,
             *   f: Array<RawFunctionSearchType>,
             *   p: Array<Object>,
             *   b: Array<[Number, String]>,
             *   c: Array<Number>
             * }}
             */
            const crateCorpus = rawSearchIndex[crate];

            searchWords.push(crate);
            // This object should have exactly the same set of fields as the "row"
            // object defined below. Your JavaScript runtime will thank you.
            // https://mathiasbynens.be/notes/shapes-ics
            const crateRow = {
                crate: crate,
                ty: 1, // == ExternCrate
                name: crate,
                path: "",
                desc: crateCorpus.doc,
                parent: undefined,
                type: null,
                id: id,
                normalizedName: crate.indexOf("_") === -1 ? crate : crate.replace(/_/g, ""),
                deprecated: null,
                implDisambiguator: null,
            };
            id += 1;
            searchIndex.push(crateRow);
            currentIndex += 1;

            // a String of one character item type codes
            const itemTypes = crateCorpus.t;
            // an array of (String) item names
            const itemNames = crateCorpus.n;
            // an array of [(Number) item index,
            //              (String) full path]
            // an item whose index is not present will fall back to the previous present path
            // i.e. if indices 4 and 11 are present, but 5-10 and 12-13 are not present,
            // 5-10 will fall back to the path for 4 and 12-13 will fall back to the path for 11
            const itemPaths = new Map(crateCorpus.q);
            // an array of (String) descriptions
            const itemDescs = crateCorpus.d;
            // an array of (Number) the parent path index + 1 to `paths`, or 0 if none
            const itemParentIdxs = crateCorpus.i;
            // an array of (Object | null) the type of the function, if any
            const itemFunctionSearchTypes = crateCorpus.f;
            // an array of (Number) indices for the deprecated items
            const deprecatedItems = new Set(crateCorpus.c);
            // an array of (Number) indices for the deprecated items
            const implDisambiguator = new Map(crateCorpus.b);
            // an array of [(Number) item type,
            //              (String) name]
            const paths = crateCorpus.p;
            // an array of [(String) alias name
            //             [Number] index to items]
            const aliases = crateCorpus.a;

            // an array of [{name: String, ty: Number}]
            const lowercasePaths = [];

            // convert `rawPaths` entries into object form
            // generate normalizedPaths for function search mode
            let len = paths.length;
            let lastPath = itemPaths.get(0);
            for (let i = 0; i < len; ++i) {
                const elem = paths[i];
                const ty = elem[0];
                const name = elem[1];
                let path = null;
                if (elem.length > 2) {
                    path = itemPaths.has(elem[2]) ? itemPaths.get(elem[2]) : lastPath;
                    lastPath = path;
                }

                lowercasePaths.push({ty: ty, name: name.toLowerCase(), path: path});
                paths[i] = {ty: ty, name: name, path: path};
            }

            // convert `item*` into an object form, and construct word indices.
            //
            // before any analysis is performed lets gather the search terms to
            // search against apart from the rest of the data.  This is a quick
            // operation that is cached for the life of the page state so that
            // all other search operations have access to this cached data for
            // faster analysis operations
            lastPath = "";
            len = itemTypes.length;
            for (let i = 0; i < len; ++i) {
                let word = "";
                // This object should have exactly the same set of fields as the "crateRow"
                // object defined above.
                if (typeof itemNames[i] === "string") {
                    word = itemNames[i].toLowerCase();
                }
                searchWords.push(word);
                const path = itemPaths.has(i) ? itemPaths.get(i) : lastPath;
                const row = {
                    crate: crate,
                    ty: itemTypes.charCodeAt(i) - charA,
                    name: itemNames[i],
                    path: path,
                    desc: itemDescs[i],
                    parent: itemParentIdxs[i] > 0 ? paths[itemParentIdxs[i] - 1] : undefined,
                    type: buildFunctionSearchType(
                        itemFunctionSearchTypes[i],
                        lowercasePaths
                    ),
                    id: id,
                    normalizedName: word.indexOf("_") === -1 ? word : word.replace(/_/g, ""),
                    deprecated: deprecatedItems.has(i),
                    implDisambiguator: implDisambiguator.has(i) ? implDisambiguator.get(i) : null,
                };
                id += 1;
                searchIndex.push(row);
                lastPath = row.path;
                crateSize += 1;
            }

            if (aliases) {
                const currentCrateAliases = new Map();
                ALIASES.set(crate, currentCrateAliases);
                for (const alias_name in aliases) {
                    if (!hasOwnPropertyRustdoc(aliases, alias_name)) {
                        continue;
                    }

                    let currentNameAliases;
                    if (currentCrateAliases.has(alias_name)) {
                        currentNameAliases = currentCrateAliases.get(alias_name);
                    } else {
                        currentNameAliases = [];
                        currentCrateAliases.set(alias_name, currentNameAliases);
                    }
                    for (const local_alias of aliases[alias_name]) {
                        currentNameAliases.push(local_alias + currentIndex);
                    }
                }
            }
            currentIndex += crateSize;
        }
        return searchWords;
    }

    /**
     * Callback for when the search form is submitted.
     * @param {Event} [e] - The event that triggered this call, if any
     */
    function onSearchSubmit(e) {
        e.preventDefault();
        searchState.clearInputTimeout();
        search();
    }

    function putBackSearch() {
        const search_input = searchState.input;
        if (!searchState.input) {
            return;
        }
        if (search_input.value !== "" && !searchState.isDisplayed()) {
            searchState.showResults();
            if (browserSupportsHistoryApi()) {
                history.replaceState(null, "",
                    buildUrl(search_input.value, getFilterCrates()));
            }
            document.title = searchState.title;
        }
    }

    function registerSearchEvents() {
        const params = searchState.getQueryStringParams();

        // Populate search bar with query string search term when provided,
        // but only if the input bar is empty. This avoid the obnoxious issue
        // where you start trying to do a search, and the index loads, and
        // suddenly your search is gone!
        if (searchState.input.value === "") {
            searchState.input.value = params.search || "";
        }

        const searchAfter500ms = () => {
            searchState.clearInputTimeout();
            if (searchState.input.value.length === 0) {
                searchState.hideResults();
            } else {
                searchState.timeout = setTimeout(search, 500);
            }
        };
        searchState.input.onkeyup = searchAfter500ms;
        searchState.input.oninput = searchAfter500ms;
        document.getElementsByClassName("search-form")[0].onsubmit = onSearchSubmit;
        searchState.input.onchange = e => {
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

        searchState.outputElement().addEventListener("keydown", e => {
            // We only handle unmodified keystrokes here. We don't want to interfere with,
            // for instance, alt-left and alt-right for history navigation.
            if (e.altKey || e.ctrlKey || e.shiftKey || e.metaKey) {
                return;
            }
            // up and down arrow select next/previous search result, or the
            // search box if we're already at the top.
            if (e.which === 38) { // up
                const previous = document.activeElement.previousElementSibling;
                if (previous) {
                    previous.focus();
                } else {
                    searchState.focus();
                }
                e.preventDefault();
            } else if (e.which === 40) { // down
                const next = document.activeElement.nextElementSibling;
                if (next) {
                    next.focus();
                }
                const rect = document.activeElement.getBoundingClientRect();
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

        searchState.input.addEventListener("keydown", e => {
            if (e.which === 40) { // down
                focusSearchResult();
                e.preventDefault();
            }
        });

        searchState.input.addEventListener("focus", () => {
            putBackSearch();
        });

        searchState.input.addEventListener("blur", () => {
            searchState.input.placeholder = searchState.input.origPlaceholder;
        });

        // Push and pop states are used to add search results to the browser
        // history.
        if (browserSupportsHistoryApi()) {
            // Store the previous <title> so we can revert back to it later.
            const previousTitle = document.title;

            window.addEventListener("popstate", e => {
                const params = searchState.getQueryStringParams();
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
        window.onpageshow = () => {
            const qSearch = searchState.getQueryStringParams().search;
            if (searchState.input.value === "" && qSearch) {
                searchState.input.value = qSearch;
            }
            search();
        };
    }

    function updateCrate(ev) {
        if (ev.target.value === "all crates") {
            // If we don't remove it from the URL, it'll be picked up again by the search.
            const query = searchState.input.value.trim();
            updateSearchHistory(buildUrl(query, null));
        }
        // In case you "cut" the entry from the search input, then change the crate filter
        // before paste back the previous search, you get the old search results without
        // the filter. To prevent this, we need to remove the previous results.
        currentResults = null;
        search(undefined, true);
    }

    /**
     *  @type {Array<string>}
     */
    const searchWords = buildIndex(rawSearchIndex);
    if (typeof window !== "undefined") {
        registerSearchEvents();
        // If there's a search term in the URL, execute the search now.
        if (window.searchState.getQueryStringParams().search) {
            search();
        }
    }

    if (typeof exports !== "undefined") {
        exports.initSearch = initSearch;
        exports.execQuery = execQuery;
        exports.parseQuery = parseQuery;
    }
    return searchWords;
}

if (typeof window !== "undefined") {
    window.initSearch = initSearch;
    if (window.searchIndex !== undefined) {
        initSearch(window.searchIndex);
    }
} else {
    // Running in Node, not a browser. Run initSearch just to produce the
    // exports.
    initSearch({});
}


})();
