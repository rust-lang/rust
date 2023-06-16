/* global addClass, getNakedUrl, getSettingValue */
/* global onEachLazy, removeClass, searchState, browserSupportsHistoryApi, exports */

"use strict";

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
    "associated type",
    "constant",
    "associated constant",
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
     * Returns the number. If name is "" or null, return -1 (pure generic).
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
            return -1;
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
        return isWhitespace(c) || isEndCharacter(c);
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
        return c === "," || isWhitespaceCharacter(c);
    }

    /**
     * Returns `true` if the given `c` character is a whitespace.
     *
     * @param {string} c
     *
     * @return {boolean}
     */
    function isWhitespaceCharacter(c) {
        return c === " " || c === "\t";
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
        if (name === "*" || (name.length === 0 && generics.length === 0)) {
            return;
        }
        if (query.literalSearch && parserState.totalElems - parserState.genericsElems > 0) {
            throw ["You cannot have more than one element if you use quotes"];
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
                id: -1,
                fullPath: ["never"],
                pathWithoutLast: [],
                pathLast: "never",
                generics: [],
                typeFilter: "primitive",
            };
        }
        const pathSegments = name.split("::");
        if (pathSegments.length > 1) {
            for (let i = 0, len = pathSegments.length; i < len; ++i) {
                const pathSegment = pathSegments[i];

                if (pathSegment.length === 0) {
                    if (i === 0) {
                        throw ["Paths cannot start with ", "::"];
                    } else if (i + 1 === len) {
                        throw ["Paths cannot end with ", "::"];
                    }
                    throw ["Unexpected ", "::::"];
                }

                if (pathSegment === "!") {
                    pathSegments[i] = "never";
                    if (i !== 0) {
                        throw ["Never type ", "!", " is not associated item"];
                    }
                }
            }
        }
        // In case we only have something like `<p>`, there is no name.
        if (pathSegments.length === 0 || (pathSegments.length === 1 && pathSegments[0] === "")) {
            throw ["Found generics without a path"];
        }
        parserState.totalElems += 1;
        if (isInGenerics) {
            parserState.genericsElems += 1;
        }
        return {
            name: name,
            id: -1,
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
                } else if (
                    isStopCharacter(c) ||
                    isSpecialStartCharacter(c) ||
                    isSeparatorCharacter(c)
                ) {
                    break;
                } else if (c === ":") { // If we allow paths ("str::string" for example).
                    if (!isPathStart(parserState)) {
                        break;
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
                    // Skip current ":".
                    parserState.pos += 1;
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
                id: -1,
                fullPath: ["[]"],
                pathWithoutLast: [],
                pathLast: "[]",
                generics,
                typeFilter: "primitive",
            });
        } else {
            // We handle the strings on their own mostly to make code easier to follow.
            if (parserState.userQuery[parserState.pos] === "\"") {
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
                    throw ["You cannot use quotes on type filter"];
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
                if (endChar !== "") {
                    throw [
                        "Expected ",
                        ",", // comma
                        ", ",
                        "&nbsp;", // whitespace
                        " or ",
                        endChar,
                        ", found ",
                        c,
                    ];
                }
                throw [
                    "Expected ",
                    ",", // comma
                    " or ",
                    "&nbsp;", // whitespace
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
        const query = parserState.userQuery;

        for (let pos = start; pos < parserState.pos; ++pos) {
            if (!isIdentCharacter(query[pos]) && !isWhitespaceCharacter(query[pos])) {
                throw ["Unexpected ", query[pos], " in type filter"];
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
                    throw ["Unexpected ", ":"];
                }
                if (query.elems.length === 0) {
                    throw ["Expected type filter before ", ":"];
                } else if (query.literalSearch) {
                    throw ["You cannot use quotes on type filter"];
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
            }
            if (!foundStopChar) {
                if (parserState.typeFilter !== null) {
                    throw [
                        "Expected ",
                        ",", // comma
                        ", ",
                        "&nbsp;", // whitespace
                        " or ",
                        "->", // arrow
                        ", found ",
                        c,
                    ];
                }
                throw [
                    "Expected ",
                    ",", // comma
                    ", ",
                    "&nbsp;", // whitespace
                    ", ",
                    ":", // colon
                    " or ",
                    "->", // arrow
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
            throw ["Unexpected ", ":", " (expected path after type filter)"];
        }
        while (parserState.pos < parserState.length) {
            if (isReturnArrow(parserState)) {
                parserState.pos += 2;
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
            literalSearch: false,
            error: null,
            correction: null,
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
     * The supported syntax by this parser is as follow:
     *
     * ident = *(ALPHA / DIGIT / "_")
     * path = ident *(DOUBLE-COLON ident) [!]
     * slice = OPEN-SQUARE-BRACKET [ nonempty-arg-list ] CLOSE-SQUARE-BRACKET
     * arg = [type-filter *WS COLON *WS] (path [generics] / slice)
     * type-sep = COMMA/WS *(COMMA/WS)
     * nonempty-arg-list = *(type-sep) arg *(type-sep arg) *(type-sep)
     * generics = OPEN-ANGLE-BRACKET [ nonempty-arg-list ] *(type-sep)
     *            CLOSE-ANGLE-BRACKET
     * return-args = RETURN-ARROW *(type-sep) nonempty-arg-list
     *
     * exact-search = [type-filter *WS COLON] [ RETURN-ARROW ] *WS QUOTE ident QUOTE [ generics ]
     * type-search = [ nonempty-arg-list ] [ return-args ]
     *
     * query = *WS (exact-search / type-search) *WS
     *
     * type-filter = (
     *     "mod" /
     *     "externcrate" /
     *     "import" /
     *     "struct" /
     *     "enum" /
     *     "fn" /
     *     "type" /
     *     "static" /
     *     "trait" /
     *     "impl" /
     *     "tymethod" /
     *     "method" /
     *     "structfield" /
     *     "variant" /
     *     "macro" /
     *     "primitive" /
     *     "associatedtype" /
     *     "constant" /
     *     "associatedconstant" /
     *     "union" /
     *     "foreigntype" /
     *     "keyword" /
     *     "existential" /
     *     "attr" /
     *     "derive" /
     *     "traitalias")
     *
     * OPEN-ANGLE-BRACKET = "<"
     * CLOSE-ANGLE-BRACKET = ">"
     * OPEN-SQUARE-BRACKET = "["
     * CLOSE-SQUARE-BRACKET = "]"
     * COLON = ":"
     * DOUBLE-COLON = "::"
     * QUOTE = %x22
     * COMMA = ","
     * RETURN-ARROW = "->"
     *
     * ALPHA = %x41-5A / %x61-7A ; A-Z / a-z
     * DIGIT = %x30-39
     * WS = %x09 / " "
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
                if (result.id > -1) {
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
         * This function checks generics in search query `queryElem` can all be found in the
         * search index (`fnType`),
         *
         * @param {FunctionType} fnType     - The object to check.
         * @param {QueryElement} queryElem  - The element from the parsed query.
         *
         * @return {boolean} - Returns true if a match, false otherwise.
         */
        function checkGenerics(fnType, queryElem) {
            return unifyFunctionTypes(fnType.generics, queryElem.generics);
        }
        /**
         * This function checks if a list of search query `queryElems` can all be found in the
         * search index (`fnTypes`).
         *
         * @param {Array<FunctionType>} fnTypes    - The objects to check.
         * @param {Array<QueryElement>} queryElems - The elements from the parsed query.
         *
         * @return {boolean} - Returns true if a match, false otherwise.
         */
        function unifyFunctionTypes(fnTypes, queryElems) {
            // This search engine implements order-agnostic unification. There
            // should be no missing duplicates (generics have "bag semantics"),
            // and the row is allowed to have extras.
            if (queryElems.length === 0) {
                return true;
            }
            if (!fnTypes || fnTypes.length === 0) {
                return false;
            }
            /**
             * @type Map<integer, QueryElement[]>
             */
            const queryElemSet = new Map();
            const addQueryElemToQueryElemSet = function addQueryElemToQueryElemSet(queryElem) {
                let currentQueryElemList;
                if (queryElemSet.has(queryElem.id)) {
                    currentQueryElemList = queryElemSet.get(queryElem.id);
                } else {
                    currentQueryElemList = [];
                    queryElemSet.set(queryElem.id, currentQueryElemList);
                }
                currentQueryElemList.push(queryElem);
            };
            for (const queryElem of queryElems) {
                addQueryElemToQueryElemSet(queryElem);
            }
            /**
             * @type Map<integer, FunctionType[]>
             */
            const fnTypeSet = new Map();
            const addFnTypeToFnTypeSet = function addFnTypeToFnTypeSet(fnType) {
                // Pure generic, or an item that's not matched by any query elems.
                // Try [unboxing] it.
                //
                // [unboxing]:
                // http://ndmitchell.com/downloads/slides-hoogle_fast_type_searching-09_aug_2008.pdf
                const queryContainsArrayOrSliceElem = queryElemSet.has(typeNameIdOfArrayOrSlice);
                if (fnType.id === -1 || !(
                    queryElemSet.has(fnType.id) ||
                    (fnType.id === typeNameIdOfSlice && queryContainsArrayOrSliceElem) ||
                    (fnType.id === typeNameIdOfArray && queryContainsArrayOrSliceElem)
                )) {
                    for (const innerFnType of fnType.generics) {
                        addFnTypeToFnTypeSet(innerFnType);
                    }
                    return;
                }
                let currentQueryElemList = queryElemSet.get(fnType.id) || [];
                let matchIdx = currentQueryElemList.findIndex(queryElem => {
                    return typePassesFilter(queryElem.typeFilter, fnType.ty) &&
                        checkGenerics(fnType, queryElem);
                });
                if (matchIdx === -1 &&
                    (fnType.id === typeNameIdOfSlice || fnType.id === typeNameIdOfArray) &&
                    queryContainsArrayOrSliceElem
                ) {
                    currentQueryElemList = queryElemSet.get(typeNameIdOfArrayOrSlice) || [];
                    matchIdx = currentQueryElemList.findIndex(queryElem => {
                        return typePassesFilter(queryElem.typeFilter, fnType.ty) &&
                            checkGenerics(fnType, queryElem);
                    });
                }
                // None of the query elems match the function type.
                // Try [unboxing] it.
                if (matchIdx === -1) {
                    for (const innerFnType of fnType.generics) {
                        addFnTypeToFnTypeSet(innerFnType);
                    }
                    return;
                }
                let currentFnTypeList;
                if (fnTypeSet.has(fnType.id)) {
                    currentFnTypeList = fnTypeSet.get(fnType.id);
                } else {
                    currentFnTypeList = [];
                    fnTypeSet.set(fnType.id, currentFnTypeList);
                }
                currentFnTypeList.push(fnType);
            };
            for (const fnType of fnTypes) {
                addFnTypeToFnTypeSet(fnType);
            }
            const doHandleQueryElemList = (currentFnTypeList, queryElemList) => {
                if (queryElemList.length === 0) {
                    return true;
                }
                // Multiple items in one list might match multiple items in another.
                // Since an item with fewer generics can match an item with more, we
                // need to check all combinations for a potential match.
                const queryElem = queryElemList.pop();
                const l = currentFnTypeList.length;
                for (let i = 0; i < l; i += 1) {
                    const fnType = currentFnTypeList[i];
                    if (!typePassesFilter(queryElem.typeFilter, fnType.ty)) {
                        continue;
                    }
                    if (queryElem.generics.length === 0 || checkGenerics(fnType, queryElem)) {
                        currentFnTypeList.splice(i, 1);
                        const result = doHandleQueryElemList(currentFnTypeList, queryElemList);
                        if (result) {
                            return true;
                        }
                        currentFnTypeList.splice(i, 0, fnType);
                    }
                }
                return false;
            };
            const handleQueryElemList = (id, queryElemList) => {
                if (!fnTypeSet.has(id)) {
                    if (id === typeNameIdOfArrayOrSlice) {
                        return handleQueryElemList(typeNameIdOfSlice, queryElemList) ||
                            handleQueryElemList(typeNameIdOfArray, queryElemList);
                    }
                    return false;
                }
                const currentFnTypeList = fnTypeSet.get(id);
                if (currentFnTypeList.length < queryElemList.length) {
                    // It's not possible for all the query elems to find a match.
                    return false;
                }
                const result = doHandleQueryElemList(currentFnTypeList, queryElemList);
                if (result) {
                    // Found a solution.
                    // Any items that weren't used for it can be unboxed, and might form
                    // part of the solution for another item.
                    for (const innerFnType of currentFnTypeList) {
                        addFnTypeToFnTypeSet(innerFnType);
                    }
                    fnTypeSet.delete(id);
                }
                return result;
            };
            let queryElemSetSize = -1;
            while (queryElemSetSize !== queryElemSet.size) {
                queryElemSetSize = queryElemSet.size;
                for (const [id, queryElemList] of queryElemSet) {
                    if (handleQueryElemList(id, queryElemList)) {
                        queryElemSet.delete(id);
                    }
                }
            }
            return queryElemSetSize === 0;
        }

        /**
          * This function checks if the object (`row`) matches the given type (`elem`) and its
          * generics (if any).
          *
          * @param {Array<FunctionType>} list
          * @param {QueryElement} elem    - The element from the parsed query.
          *
          * @return {boolean} - Returns true if found, false otherwise.
          */
        function checkIfInList(list, elem) {
            for (const entry of list) {
                if (checkType(entry, elem)) {
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
          * @param {QueryElement} elem      - The element from the parsed query.
          *
          * @return {boolean} - Returns true if the type matches, false otherwise.
          */
        function checkType(row, elem) {
            if (row.id === -1) {
                // This is a pure "generic" search, no need to run other checks.
                return row.generics.length > 0 ? checkIfInList(row.generics, elem) : false;
            }

            const matchesExact = row.id === elem.id;
            const matchesArrayOrSlice = elem.id === typeNameIdOfArrayOrSlice &&
                (row.id === typeNameIdOfSlice || row.id === typeNameIdOfArray);

            if ((matchesExact || matchesArrayOrSlice) &&
                typePassesFilter(elem.typeFilter, row.ty)) {
                if (elem.generics.length > 0) {
                    return checkGenerics(row, elem);
                }
                return true;
            }

            // If the current item does not match, try [unboxing] the generic.
            // [unboxing]:
            //   https://ndmitchell.com/downloads/slides-hoogle_fast_type_searching-09_aug_2008.pdf
            return checkIfInList(row.generics, elem);
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

            const in_args = row.type && row.type.inputs && checkIfInList(row.type.inputs, elem);
            if (in_args) {
                // path_dist is 0 because no parent path information is currently stored
                // in the search index
                addIntoResults(results_in_args, fullId, pos, -1, 0, 0, maxEditDistance);
            }
            const returned = row.type && row.type.output && checkIfInList(row.type.output, elem);
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
            if (!unifyFunctionTypes(row.type.inputs, parsedQuery.elems)) {
                return;
            }
            if (!unifyFunctionTypes(row.type.output, parsedQuery.returned)) {
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
                if (typeNameIdMap.has(elem.name)) {
                    elem.id = typeNameIdMap.get(elem.name);
                } else if (!parsedQuery.literalSearch) {
                    let match = -1;
                    let matchDist = maxEditDistance + 1;
                    let matchName = "";
                    for (const [name, id] of typeNameIdMap) {
                        const dist = editDistance(name, elem.name, maxEditDistance);
                        if (dist <= matchDist && dist <= maxEditDistance) {
                            if (dist === matchDist && matchName > name) {
                                continue;
                            }
                            match = id;
                            matchDist = dist;
                            matchName = name;
                        }
                    }
                    if (match !== -1) {
                        parsedQuery.correction = matchName;
                    }
                    elem.id = match;
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
                        in_returned = row.type &&
                            unifyFunctionTypes(row.type.output, parsedQuery.returned);
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
            let anchor = "#" + type + "." + name;
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
                anchor = "#variant." + myparent.name + ".field." + name;
                pageType = "enum";
                pageName = enumName;
            } else {
                displayPath = path + "::" + myparent.name + "::";
            }
            href = ROOT_PATH + path.replace(/::/g, "/") +
                "/" + pageType +
                "." + pageName +
                ".html" + anchor;
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

                if (item.is_alias) {
                    const alias = document.createElement("span");
                    alias.className = "alias";

                    const bold = document.createElement("b");
                    bold.innerText = item.alias;
                    alias.appendChild(bold);

                    alias.insertAdjacentHTML(
                        "beforeend",
                        "<i class=\"grey\">&nbsp;- see&nbsp;</i>");

                    resultName.appendChild(alias);
                }

                resultName.insertAdjacentHTML(
                    "beforeend",
                    `${typeName} ${item.displayPath}<span class="${type}">${name}</span>`);
                link.appendChild(resultName);

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
        if (searchState.currentTab === tabNb) {
            return "<button class=\"selected\">" + text +
                   " <span class=\"count\">(" + nbElems + ")</span></button>";
        }
        return "<button>" + text + " <span class=\"count\">(" + nbElems + ")</span></button>";
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
                    error[index] = `<code>${value}</code>`;
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
        if (browserSupportsHistoryApi()) {
            const newURL = buildUrl(query.original, filterCrates);

            if (!history.state && !params.search) {
                history.pushState(null, "", newURL);
            } else {
                history.replaceState(null, "", newURL);
            }
        }

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
        const PATH_INDEX_DATA = 0;
        const GENERICS_DATA = 1;
        return types.map(type => {
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
            return {
                // `0` is used as a sentinel because it's fewer bytes than `null`
                id: pathIndex === 0
                    ? -1
                    : buildTypeMapIndex(lowercasePaths[pathIndex - 1].name),
                ty: pathIndex === 0 ? null : lowercasePaths[pathIndex - 1].ty,
                generics: generics,
            };
        });
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
            const pathIndex = functionSearchType[INPUTS_DATA];
            inputs = [{
                id: pathIndex === 0
                    ? -1
                    : buildTypeMapIndex(lowercasePaths[pathIndex - 1].name),
                ty: pathIndex === 0 ? null : lowercasePaths[pathIndex - 1].ty,
                generics: [],
            }];
        } else {
            inputs = buildItemSearchTypeAll(
                functionSearchType[INPUTS_DATA],
                lowercasePaths
            );
        }
        if (functionSearchType.length > 1) {
            if (typeof functionSearchType[OUTPUT_DATA] === "number") {
                const pathIndex = functionSearchType[OUTPUT_DATA];
                output = [{
                    id: pathIndex === 0
                        ? -1
                        : buildTypeMapIndex(lowercasePaths[pathIndex - 1].name),
                    ty: pathIndex === 0 ? null : lowercasePaths[pathIndex - 1].ty,
                    generics: [],
                }];
            } else {
                output = buildItemSearchTypeAll(
                    functionSearchType[OUTPUT_DATA],
                    lowercasePaths
                );
            }
        } else {
            output = [];
        }
        return {
            inputs, output,
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
             * The raw search data for a given crate. `n`, `t`, `d`, and `q`, `i`, and `f`
             * are arrays with the same length. n[i] contains the name of an item.
             * t[i] contains the type of that item (as a string of characters that represent an
             * offset in `itemTypes`). d[i] contains the description of that item.
             *
             * q[i] contains the full path of the item, or an empty string indicating
             * "same as q[i-1]".
             *
             * i[i] contains an item's parent, usually a module. For compactness,
             * it is a set of indexes into the `p` array.
             *
             * f[i] contains function signatures, or `0` if the item isn't a function.
             * Functions are themselves encoded as arrays. The first item is a list of
             * types representing the function's inputs, and the second list item is a list
             * of types representing the function's output. Tuples are flattened.
             * Types are also represented as arrays; the first item is an index into the `p`
             * array, while the second is a list of types representing any generic parameters.
             *
             * `a` defines aliases with an Array of pairs: [name, offset], where `offset`
             * points into the n/t/d/q/i/f arrays.
             *
             * `doc` contains the description of the crate.
             *
             * `p` is a list of path/type pairs. It is used for parents and function parameters.
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
            for (let i = 0; i < len; ++i) {
                lowercasePaths.push({ty: paths[i][0], name: paths[i][1].toLowerCase()});
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
            let lastPath = "";
            for (let i = 0; i < len; ++i) {
                let word = "";
                // This object should have exactly the same set of fields as the "crateRow"
                // object defined above.
                if (typeof itemNames[i] === "string") {
                    word = itemNames[i].toLowerCase();
                }
                searchWords.push(word);
                const row = {
                    crate: crate,
                    ty: itemTypes.charCodeAt(i) - charA,
                    name: itemNames[i],
                    path: itemPaths.has(i) ? itemPaths.get(i) : lastPath,
                    desc: itemDescs[i],
                    parent: itemParentIdxs[i] > 0 ? paths[itemParentIdxs[i] - 1] : undefined,
                    type: buildFunctionSearchType(
                        itemFunctionSearchTypes[i],
                        lowercasePaths
                    ),
                    id: id,
                    normalizedName: word.indexOf("_") === -1 ? word : word.replace(/_/g, ""),
                    deprecated: deprecatedItems.has(i),
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
            const params = searchState.getQueryStringParams();
            const query = searchState.input.value.trim();
            if (!history.state && !params.search) {
                history.pushState(null, "", buildUrl(query, null));
            } else {
                history.replaceState(null, "", buildUrl(query, null));
            }
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
