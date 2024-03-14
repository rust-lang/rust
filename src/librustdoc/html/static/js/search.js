// ignore-tidy-filelength
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
    "keyword",
    "primitive",
    "mod",
    "externcrate",
    "import",
    "struct", // 5
    "enum",
    "fn",
    "type",
    "static",
    "trait", // 10
    "impl",
    "tymethod",
    "method",
    "structfield",
    "variant", // 15
    "macro",
    "associatedtype",
    "constant",
    "associatedconstant",
    "union", // 20
    "foreigntype",
    "existential",
    "attr",
    "derive",
    "traitalias", // 25
    "generic",
];

const longItemTypes = [
    "keyword",
    "primitive type",
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
    "assoc type",
    "constant",
    "assoc const",
    "union",
    "foreign type",
    "existential type",
    "attribute macro",
    "derive macro",
    "trait alias",
];

// used for special search precedence
const TY_GENERIC = itemTypes.indexOf("generic");
const ROOT_PATH = typeof window !== "undefined" ? window.rootPath : "../";

// Hard limit on how deep to recurse into generics when doing type-driven search.
// This needs limited, partially because
// a search for `Ty` shouldn't match `WithInfcx<ParamEnvAnd<Vec<ConstTy<Interner<Ty=Ty>>>>>`,
// but mostly because this is the simplest and most principled way to limit the number
// of permutations we need to check.
const UNBOXING_LIMIT = 5;

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
    /**
     *  @type {Uint32Array}
     */
    let functionTypeFingerprint;
    let currentResults;
    /**
     * Map from normalized type names to integers. Used to make type search
     * more efficient.
     *
     * @type {Map<string, {id: integer, assocOnly: boolean}>}
     */
    const typeNameIdMap = new Map();
    const ALIASES = new Map();

    /**
     * Special type name IDs for searching by array.
     */
    const typeNameIdOfArray = buildTypeMapIndex("array");
    /**
     * Special type name IDs for searching by slice.
     */
    const typeNameIdOfSlice = buildTypeMapIndex("slice");
    /**
     * Special type name IDs for searching by both array and slice (`[]` syntax).
     */
    const typeNameIdOfArrayOrSlice = buildTypeMapIndex("[]");
    /**
     * Special type name IDs for searching by tuple.
     */
    const typeNameIdOfTuple = buildTypeMapIndex("tuple");
    /**
     * Special type name IDs for searching by unit.
     */
    const typeNameIdOfUnit = buildTypeMapIndex("unit");
    /**
     * Special type name IDs for searching by both tuple and unit (`()` syntax).
     */
    const typeNameIdOfTupleOrUnit = buildTypeMapIndex("()");
    /**
     * Special type name IDs for searching `fn`.
     */
    const typeNameIdOfFn = buildTypeMapIndex("fn");
    /**
     * Special type name IDs for searching `fnmut`.
     */
    const typeNameIdOfFnMut = buildTypeMapIndex("fnmut");
    /**
     * Special type name IDs for searching `fnonce`.
     */
    const typeNameIdOfFnOnce = buildTypeMapIndex("fnonce");
    /**
     * Special type name IDs for searching higher order functions (`->` syntax).
     */
    const typeNameIdOfHof = buildTypeMapIndex("->");

    /**
     * Add an item to the type Name->ID map, or, if one already exists, use it.
     * Returns the number. If name is "" or null, return null (pure generic).
     *
     * This is effectively string interning, so that function matching can be
     * done more quickly. Two types with the same name but different item kinds
     * get the same ID.
     *
     * @param {string} name
     * @param {boolean} isAssocType - True if this is an assoc type
     *
     * @returns {integer}
     */
    function buildTypeMapIndex(name, isAssocType) {
        if (name === "" || name === null) {
            return null;
        }

        if (typeNameIdMap.has(name)) {
            const obj = typeNameIdMap.get(name);
            obj.assocOnly = isAssocType && obj.assocOnly;
            return obj.id;
        } else {
            const id = typeNameIdMap.size;
            typeNameIdMap.set(name, {id, assocOnly: isAssocType});
            return id;
        }
    }

    function isSpecialStartCharacter(c) {
        return "<\"".indexOf(c) !== -1;
    }

    function isEndCharacter(c) {
        return "=,>-])".indexOf(c) !== -1;
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
        return c === "," || c === "=";
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
        return c === ":" || c === " ";
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
            } else if (c !== " ") {
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
            if (c !== " ") {
                break;
            }
            parserState.pos += 1;
        }
    }

    function makePrimitiveElement(name, extra) {
        return Object.assign({
            name,
            id: null,
            fullPath: [name],
            pathWithoutLast: [],
            pathLast: name,
            normalizedPathLast: name,
            generics: [],
            bindings: new Map(),
            typeFilter: "primitive",
            bindingName: null,
        }, extra);
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
            const bindingName = parserState.isInBinding;
            parserState.isInBinding = null;
            return makePrimitiveElement("never", { bindingName });
        }
        const quadcolon = /::\s*::/.exec(path);
        if (path.startsWith("::")) {
            throw ["Paths cannot start with ", "::"];
        } else if (path.endsWith("::")) {
            throw ["Paths cannot end with ", "::"];
        } else if (quadcolon !== null) {
            throw ["Unexpected ", quadcolon[0]];
        }
        const pathSegments = path.split(/(?:::\s*)|(?:\s+(?:::\s*)?)/);
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
        const bindingName = parserState.isInBinding;
        parserState.isInBinding = null;
        const bindings = new Map();
        const pathLast = pathSegments[pathSegments.length - 1];
        return {
            name: name.trim(),
            id: null,
            fullPath: pathSegments,
            pathWithoutLast: pathSegments.slice(0, pathSegments.length - 1),
            pathLast,
            normalizedPathLast: pathLast.replace(/_/g, ""),
            generics: generics.filter(gen => {
                // Syntactically, bindings are parsed as generics,
                // but the query engine treats them differently.
                if (gen.bindingName !== null) {
                    if (gen.name !== null) {
                        gen.bindingName.generics.unshift(gen);
                    }
                    bindings.set(gen.bindingName.name, gen.bindingName.generics);
                    return false;
                }
                return true;
            }),
            bindings,
            typeFilter,
            bindingName,
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
                            if (next_c !== " ") {
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
                    c === "(" ||
                    isEndCharacter(c) ||
                    isSpecialStartCharacter(c) ||
                    isSeparatorCharacter(c)
                ) {
                    break;
                } else if (parserState.pos > 0) {
                    throw ["Unexpected ", c, " after ", parserState.userQuery[parserState.pos - 1]];
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

    function getFilteredNextElem(query, parserState, elems, isInGenerics) {
        const start = parserState.pos;
        if (parserState.userQuery[parserState.pos] === ":" && !isPathStart(parserState)) {
            throw ["Expected type filter before ", ":"];
        }
        getNextElem(query, parserState, elems, isInGenerics);
        if (parserState.userQuery[parserState.pos] === ":" && !isPathStart(parserState)) {
            if (parserState.typeFilter !== null) {
                throw [
                    "Unexpected ",
                    ":",
                    " (expected path after type filter ",
                    parserState.typeFilter + ":",
                    ")",
                ];
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
            getNextElem(query, parserState, elems, isInGenerics);
        }
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
        if ("[(".indexOf(parserState.userQuery[parserState.pos]) !== -1) {
            let endChar = ")";
            let name = "()";
            let friendlyName = "tuple";

            if (parserState.userQuery[parserState.pos] === "[") {
                endChar = "]";
                name = "[]";
                friendlyName = "slice";
            }
            parserState.pos += 1;
            const { foundSeparator } = getItemsBefore(query, parserState, generics, endChar);
            const typeFilter = parserState.typeFilter;
            const bindingName = parserState.isInBinding;
            parserState.typeFilter = null;
            parserState.isInBinding = null;
            for (const gen of generics) {
                if (gen.bindingName !== null) {
                    throw ["Type parameter ", "=", ` cannot be within ${friendlyName} `, name];
                }
            }
            if (name === "()" && !foundSeparator && generics.length === 1 && typeFilter === null) {
                elems.push(generics[0]);
            } else if (name === "()" && generics.length === 1 && generics[0].name === "->") {
                // `primitive:(a -> b)` parser to `primitive:"->"<output=b, (a,)>`
                // not `primitive:"()"<"->"<output=b, (a,)>>`
                generics[0].typeFilter = typeFilter;
                elems.push(generics[0]);
            } else {
                if (typeFilter !== null && typeFilter !== "primitive") {
                    throw [
                        "Invalid search type: primitive ",
                        name,
                        " and ",
                        typeFilter,
                        " both specified",
                    ];
                }
                parserState.totalElems += 1;
                if (isInGenerics) {
                    parserState.genericsElems += 1;
                }
                elems.push(makePrimitiveElement(name, { bindingName, generics }));
            }
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
            } else if (parserState.pos < parserState.length &&
                parserState.userQuery[parserState.pos] === "("
            ) {
                if (start >= end) {
                    throw ["Found generics without a path"];
                }
                if (parserState.isInBinding) {
                    throw ["Unexpected ", "(", " after ", "="];
                }
                parserState.pos += 1;
                const typeFilter = parserState.typeFilter;
                parserState.typeFilter = null;
                getItemsBefore(query, parserState, generics, ")");
                skipWhitespace(parserState);
                if (isReturnArrow(parserState)) {
                    parserState.pos += 2;
                    skipWhitespace(parserState);
                    getFilteredNextElem(query, parserState, generics, isInGenerics);
                    generics[generics.length - 1].bindingName = makePrimitiveElement("output");
                } else {
                    generics.push(makePrimitiveElement(null, {
                        bindingName: makePrimitiveElement("output"),
                        typeFilter: null,
                    }));
                }
                parserState.typeFilter = typeFilter;
            }
            if (isStringElem) {
                skipWhitespace(parserState);
            }
            if (start >= end && generics.length === 0) {
                return;
            }
            if (parserState.userQuery[parserState.pos] === "=") {
                if (parserState.isInBinding) {
                    throw ["Cannot write ", "=", " twice in a binding"];
                }
                if (!isInGenerics) {
                    throw ["Type parameter ", "=", " must be within generics list"];
                }
                const name = parserState.userQuery.slice(start, end).trim();
                if (name === "!") {
                    throw ["Type parameter ", "=", " key cannot be ", "!", " never type"];
                }
                if (name.includes("!")) {
                    throw ["Type parameter ", "=", " key cannot be ", "!", " macro"];
                }
                if (name.includes("::")) {
                    throw ["Type parameter ", "=", " key cannot contain ", "::", " path"];
                }
                if (name.includes(":")) {
                    throw ["Type parameter ", "=", " key cannot contain ", ":", " type"];
                }
                parserState.isInBinding = { name, generics };
            } else {
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
     * @returns {{foundSeparator: bool}}
     */
    function getItemsBefore(query, parserState, elems, endChar) {
        let foundStopChar = true;
        let foundSeparator = false;

        // If this is a generic, keep the outer item's type filter around.
        const oldTypeFilter = parserState.typeFilter;
        parserState.typeFilter = null;
        const oldIsInBinding = parserState.isInBinding;
        parserState.isInBinding = null;

        // ML-style Higher Order Function notation
        //
        // a way to search for any closure or fn pointer regardless of
        // which closure trait is used
        //
        // Looks like this:
        //
        //     `option<t>, (t -> u) -> option<u>`
        //                  ^^^^^^
        //
        // The Rust-style closure notation is implemented in getNextElem
        let hofParameters = null;

        let extra = "";
        if (endChar === ">") {
            extra = "<";
        } else if (endChar === "]") {
            extra = "[";
        } else if (endChar === ")") {
            extra = "(";
        } else if (endChar === "") {
            extra = "->";
        } else {
            extra = endChar;
        }

        while (parserState.pos < parserState.length) {
            const c = parserState.userQuery[parserState.pos];
            if (c === endChar) {
                if (parserState.isInBinding) {
                    throw ["Unexpected ", endChar, " after ", "="];
                }
                break;
            } else if (endChar !== "" && isReturnArrow(parserState)) {
                // ML-style HOF notation only works when delimited in something,
                // otherwise a function arrow starts the return type of the top
                if (parserState.isInBinding) {
                    throw ["Unexpected ", "->", " after ", "="];
                }
                hofParameters = [...elems];
                elems.length = 0;
                parserState.pos += 2;
                foundStopChar = true;
                foundSeparator = false;
                continue;
            } else if (c === " ") {
                parserState.pos += 1;
                continue;
            } else if (isSeparatorCharacter(c)) {
                parserState.pos += 1;
                foundStopChar = true;
                foundSeparator = true;
                continue;
            } else if (c === ":" && isPathStart(parserState)) {
                throw ["Unexpected ", "::", ": paths cannot start with ", "::"];
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
                        ", ",
                        "=",
                        ", or ",
                        endChar,
                        ...extra,
                        ", found ",
                        c,
                    ];
                }
                throw [
                    "Expected ",
                    ",",
                    " or ",
                    "=",
                    ...extra,
                    ", found ",
                    c,
                ];
            }
            const posBefore = parserState.pos;
            getFilteredNextElem(query, parserState, elems, endChar !== "");
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

        if (hofParameters) {
            // Commas in a HOF don't cause wrapping parens to become a tuple.
            // If you want a one-tuple with a HOF in it, write `((a -> b),)`.
            foundSeparator = false;
            // HOFs can't have directly nested bindings.
            if ([...elems, ...hofParameters].some(x => x.bindingName) || parserState.isInBinding) {
                throw ["Unexpected ", "=", " within ", "->"];
            }
            // HOFs are represented the same way closures are.
            // The arguments are wrapped in a tuple, and the output
            // is a binding, even though the compiler doesn't technically
            // represent fn pointers that way.
            const hofElem = makePrimitiveElement("->", {
                generics: hofParameters,
                bindings: new Map([["output", [...elems]]]),
                typeFilter: null,
            });
            elems.length = 0;
            elems[0] = hofElem;
        }

        parserState.typeFilter = oldTypeFilter;
        parserState.isInBinding = oldIsInBinding;

        return { foundSeparator };
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

        while (parserState.pos < parserState.length) {
            const c = parserState.userQuery[parserState.pos];
            if (isEndCharacter(c)) {
                foundStopChar = true;
                if (isSeparatorCharacter(c)) {
                    parserState.pos += 1;
                    continue;
                } else if (c === "-" || c === ">") {
                    if (isReturnArrow(parserState)) {
                        break;
                    }
                    throw ["Unexpected ", c, " (did you mean ", "->", "?)"];
                } else if (parserState.pos > 0) {
                    throw ["Unexpected ", c, " after ", parserState.userQuery[parserState.pos - 1]];
                }
                throw ["Unexpected ", c];
            } else if (c === " ") {
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
            getFilteredNextElem(query, parserState, query.elems, false);
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
            // bloom filter build from type ids
            typeFingerprint: new Uint32Array(4),
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
            rawSearchIndex.has(elem.value)
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
            for (const constraints of elem.bindings.values()) {
                for (const constraint of constraints) {
                    convertTypeFilterOnElem(constraint);
                }
            }
        }
        userQuery = userQuery.trim().replace(/\r|\n|\t/g, " ");
        const parserState = {
            length: userQuery.length,
            pos: 0,
            // Total number of elements (includes generics).
            totalElems: 0,
            genericsElems: 0,
            typeFilter: null,
            isInBinding: null,
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
     * @param  {Object} [filterCrates]   - Crate to search in if defined
     * @param  {Object} [currentCrate]   - Current crate, to rank results from this crate higher
     *
     * @return {ResultsTable}
     */
    function execQuery(parsedQuery, filterCrates, currentCrate) {
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
            const userQuery = parsedQuery.userQuery;
            const result_list = [];
            for (const result of results.values()) {
                result.item = searchIndex[result.id];
                result.word = searchIndex[result.id].word;
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
         * @param {Array<FunctionType>} fnTypesIn - The objects to check.
         * @param {Array<QueryElement>} queryElems - The elements from the parsed query.
         * @param {[FunctionType]} whereClause - Trait bounds for generic items.
         * @param {Map<number,number>|null} mgensIn
         *     - Map functions generics to query generics (never modified).
         * @param {null|Map<number,number> -> bool} solutionCb - Called for each `mgens` solution.
         * @param {number} unboxingDepth
         *     - Limit checks that Ty matches Vec<Ty>,
         *       but not Vec<ParamEnvAnd<WithInfcx<ConstTy<Interner<Ty=Ty>>>>>
         *
         * @return {boolean} - Returns true if a match, false otherwise.
         */
        function unifyFunctionTypes(
            fnTypesIn,
            queryElems,
            whereClause,
            mgensIn,
            solutionCb,
            unboxingDepth
        ) {
            if (unboxingDepth >= UNBOXING_LIMIT) {
                return false;
            }
            /**
             * @type Map<integer, integer>|null
             */
            const mgens = mgensIn === null ? null : new Map(mgensIn);
            if (queryElems.length === 0) {
                return !solutionCb || solutionCb(mgens);
            }
            if (!fnTypesIn || fnTypesIn.length === 0) {
                return false;
            }
            const ql = queryElems.length;
            const fl = fnTypesIn.length;

            // One element fast path / base case
            if (ql === 1 && queryElems[0].generics.length === 0
                && queryElems[0].bindings.size === 0) {
                const queryElem = queryElems[0];
                for (const fnType of fnTypesIn) {
                    if (!unifyFunctionTypeIsMatchCandidate(fnType, queryElem, mgens)) {
                        continue;
                    }
                    if (fnType.id < 0 && queryElem.id < 0) {
                        if (mgens && mgens.has(fnType.id) &&
                            mgens.get(fnType.id) !== queryElem.id) {
                            continue;
                        }
                        const mgensScratch = new Map(mgens);
                        mgensScratch.set(fnType.id, queryElem.id);
                        if (!solutionCb || solutionCb(mgensScratch)) {
                            return true;
                        }
                    } else if (!solutionCb || solutionCb(mgens ? new Map(mgens) : null)) {
                        // unifyFunctionTypeIsMatchCandidate already checks that ids match
                        return true;
                    }
                }
                for (const fnType of fnTypesIn) {
                    if (!unifyFunctionTypeIsUnboxCandidate(
                        fnType,
                        queryElem,
                        whereClause,
                        mgens,
                        unboxingDepth + 1
                    )) {
                        continue;
                    }
                    if (fnType.id < 0) {
                        if (mgens && mgens.has(fnType.id) &&
                            mgens.get(fnType.id) !== 0) {
                            continue;
                        }
                        const mgensScratch = new Map(mgens);
                        mgensScratch.set(fnType.id, 0);
                        if (unifyFunctionTypes(
                            whereClause[(-fnType.id) - 1],
                            queryElems,
                            whereClause,
                            mgensScratch,
                            solutionCb,
                            unboxingDepth + 1
                        )) {
                            return true;
                        }
                    } else if (unifyFunctionTypes(
                        [...fnType.generics, ...Array.from(fnType.bindings.values()).flat() ],
                        queryElems,
                        whereClause,
                        mgens ? new Map(mgens) : null,
                        solutionCb,
                        unboxingDepth + 1
                    )) {
                        return true;
                    }
                }
                return false;
            }

            // Multiple element recursive case
            /**
             * @type Array<FunctionType>
             */
            const fnTypes = fnTypesIn.slice();
            /**
             * Algorithm works by building up a solution set in the working arrays
             * fnTypes gets mutated in place to make this work, while queryElems
             * is left alone.
             *
             * It works backwards, because arrays can be cheaply truncated that way.
             *
             *                         vvvvvvv `queryElem`
             * queryElems = [ unknown, unknown, good, good, good ]
             * fnTypes    = [ unknown, unknown, good, good, good ]
             *                ^^^^^^^^^^^^^^^^ loop over these elements to find candidates
             *
             * Everything in the current working solution is known to be a good
             * match, but it might not be the match we wind up going with, because
             * there might be more than one candidate match, and we need to try them all
             * before giving up. So, to handle this, it backtracks on failure.
             */
            const flast = fl - 1;
            const qlast = ql - 1;
            const queryElem = queryElems[qlast];
            let queryElemsTmp = null;
            for (let i = flast; i >= 0; i -= 1) {
                const fnType = fnTypes[i];
                if (!unifyFunctionTypeIsMatchCandidate(fnType, queryElem, mgens)) {
                    continue;
                }
                let mgensScratch;
                if (fnType.id < 0) {
                    mgensScratch = new Map(mgens);
                    if (mgensScratch.has(fnType.id)
                        && mgensScratch.get(fnType.id) !== queryElem.id) {
                        continue;
                    }
                    mgensScratch.set(fnType.id, queryElem.id);
                } else {
                    mgensScratch = mgens;
                }
                // fnTypes[i] is a potential match
                // fnTypes[flast] is the last item in the list
                // swap them, and drop the potential match from the list
                // check if the remaining function types also match
                fnTypes[i] = fnTypes[flast];
                fnTypes.length = flast;
                if (!queryElemsTmp) {
                    queryElemsTmp = queryElems.slice(0, qlast);
                }
                const passesUnification = unifyFunctionTypes(
                    fnTypes,
                    queryElemsTmp,
                    whereClause,
                    mgensScratch,
                    mgensScratch => {
                        if (fnType.generics.length === 0 && queryElem.generics.length === 0
                            && fnType.bindings.size === 0 && queryElem.bindings.size === 0) {
                            return !solutionCb || solutionCb(mgensScratch);
                        }
                        const solution = unifyFunctionTypeCheckBindings(
                            fnType,
                            queryElem,
                            whereClause,
                            mgensScratch,
                            unboxingDepth
                        );
                        if (!solution) {
                            return false;
                        }
                        const simplifiedGenerics = solution.simplifiedGenerics;
                        for (const simplifiedMgens of solution.mgens) {
                            const passesUnification = unifyFunctionTypes(
                                simplifiedGenerics,
                                queryElem.generics,
                                whereClause,
                                simplifiedMgens,
                                solutionCb,
                                unboxingDepth
                            );
                            if (passesUnification) {
                                return true;
                            }
                        }
                        return false;
                    },
                    unboxingDepth
                );
                if (passesUnification) {
                    return true;
                }
                // backtrack
                fnTypes[flast] = fnTypes[i];
                fnTypes[i] = fnType;
                fnTypes.length = fl;
            }
            for (let i = flast; i >= 0; i -= 1) {
                const fnType = fnTypes[i];
                if (!unifyFunctionTypeIsUnboxCandidate(
                    fnType,
                    queryElem,
                    whereClause,
                    mgens,
                    unboxingDepth + 1
                )) {
                    continue;
                }
                let mgensScratch;
                if (fnType.id < 0) {
                    mgensScratch = new Map(mgens);
                    if (mgensScratch.has(fnType.id) && mgensScratch.get(fnType.id) !== 0) {
                        continue;
                    }
                    mgensScratch.set(fnType.id, 0);
                } else {
                    mgensScratch = mgens;
                }
                const generics = fnType.id < 0 ?
                    whereClause[(-fnType.id) - 1] :
                    fnType.generics;
                const bindings = fnType.bindings ?
                    Array.from(fnType.bindings.values()).flat() :
                    [];
                const passesUnification = unifyFunctionTypes(
                    fnTypes.toSpliced(i, 1, ...generics, ...bindings),
                    queryElems,
                    whereClause,
                    mgensScratch,
                    solutionCb,
                    unboxingDepth + 1
                );
                if (passesUnification) {
                    return true;
                }
            }
            return false;
        }
        /**
         * Check if this function is a match candidate.
         *
         * This function is all the fast checks that don't require backtracking.
         * It checks that two items are not named differently, and is load-bearing for that.
         * It also checks that, if the query has generics, the function type must have generics
         * or associated type bindings: that's not load-bearing, but it prevents unnecessary
         * backtracking later.
         *
         * @param {FunctionType} fnType
         * @param {QueryElement} queryElem
         * @param {Map<number,number>|null} mgensIn - Map functions generics to query generics.
         * @returns {boolean}
         */
        function unifyFunctionTypeIsMatchCandidate(fnType, queryElem, mgensIn) {
            // type filters look like `trait:Read` or `enum:Result`
            if (!typePassesFilter(queryElem.typeFilter, fnType.ty)) {
                return false;
            }
            // fnType.id < 0 means generic
            // queryElem.id < 0 does too
            // mgensIn[fnType.id] = queryElem.id
            // or, if mgensIn[fnType.id] = 0, then we've matched this generic with a bare trait
            // and should make that same decision everywhere it appears
            if (fnType.id < 0 && queryElem.id < 0) {
                if (mgensIn) {
                    if (mgensIn.has(fnType.id) && mgensIn.get(fnType.id) !== queryElem.id) {
                        return false;
                    }
                    for (const [fid, qid] of mgensIn.entries()) {
                        if (fnType.id !== fid && queryElem.id === qid) {
                            return false;
                        }
                        if (fnType.id === fid && queryElem.id !== qid) {
                            return false;
                        }
                    }
                }
                return true;
            } else {
                if (queryElem.id === typeNameIdOfArrayOrSlice &&
                    (fnType.id === typeNameIdOfSlice || fnType.id === typeNameIdOfArray)
                ) {
                    // [] matches primitive:array or primitive:slice
                    // if it matches, then we're fine, and this is an appropriate match candidate
                } else if (queryElem.id === typeNameIdOfTupleOrUnit &&
                    (fnType.id === typeNameIdOfTuple || fnType.id === typeNameIdOfUnit)
                ) {
                    // () matches primitive:tuple or primitive:unit
                    // if it matches, then we're fine, and this is an appropriate match candidate
                } else if (queryElem.id === typeNameIdOfHof &&
                    (fnType.id === typeNameIdOfFn || fnType.id === typeNameIdOfFnMut ||
                        fnType.id === typeNameIdOfFnOnce)
                ) {
                    // -> matches fn, fnonce, and fnmut
                    // if it matches, then we're fine, and this is an appropriate match candidate
                } else if (fnType.id !== queryElem.id || queryElem.id === null) {
                    return false;
                }
                // If the query elem has generics, and the function doesn't,
                // it can't match.
                if ((fnType.generics.length + fnType.bindings.size) === 0 &&
                    queryElem.generics.length !== 0
                ) {
                    return false;
                }
                if (fnType.bindings.size < queryElem.bindings.size) {
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
                return true;
            }
        }
        /**
         * This function checks the associated type bindings. Any that aren't matched get converted
         * to generics, and this function returns an array of the function's generics with these
         * simplified bindings added to them. That is, it takes a path like this:
         *
         *     Iterator<Item=u32>
         *
         * ... if queryElem itself has an `Item=` in it, then this function returns an empty array.
         * But if queryElem contains no Item=, then this function returns a one-item array with the
         * ID of u32 in it, and the rest of the matching engine acts as if `Iterator<u32>` were
         * the type instead.
         *
         * @param {FunctionType} fnType
         * @param {QueryElement} queryElem
         * @param {[FunctionType]} whereClause - Trait bounds for generic items.
         * @param {Map<number,number>} mgensIn - Map functions generics to query generics.
         *                                            Never modified.
         * @param {number} unboxingDepth
         * @returns {false|{mgens: [Map<number,number>], simplifiedGenerics: [FunctionType]}}
         */
        function unifyFunctionTypeCheckBindings(
            fnType,
            queryElem,
            whereClause,
            mgensIn,
            unboxingDepth
        ) {
            if (fnType.bindings.size < queryElem.bindings.size) {
                return false;
            }
            let simplifiedGenerics = fnType.generics || [];
            if (fnType.bindings.size > 0) {
                let mgensSolutionSet = [mgensIn];
                for (const [name, constraints] of queryElem.bindings.entries()) {
                    if (mgensSolutionSet.length === 0) {
                        return false;
                    }
                    if (!fnType.bindings.has(name)) {
                        return false;
                    }
                    const fnTypeBindings = fnType.bindings.get(name);
                    mgensSolutionSet = mgensSolutionSet.flatMap(mgens => {
                        const newSolutions = [];
                        unifyFunctionTypes(
                            fnTypeBindings,
                            constraints,
                            whereClause,
                            mgens,
                            newMgens => {
                                newSolutions.push(newMgens);
                                // return `false` makes unifyFunctionTypes return the full set of
                                // possible solutions
                                return false;
                            },
                            unboxingDepth
                        );
                        return newSolutions;
                    });
                }
                if (mgensSolutionSet.length === 0) {
                    return false;
                }
                const binds = Array.from(fnType.bindings.entries()).flatMap(entry => {
                    const [name, constraints] = entry;
                    if (queryElem.bindings.has(name)) {
                        return [];
                    } else {
                        return constraints;
                    }
                });
                if (simplifiedGenerics.length > 0) {
                    simplifiedGenerics = [...simplifiedGenerics, ...binds];
                } else {
                    simplifiedGenerics = binds;
                }
                return { simplifiedGenerics, mgens: mgensSolutionSet };
            }
            return { simplifiedGenerics, mgens: [mgensIn] };
        }
        /**
         * @param {FunctionType} fnType
         * @param {QueryElement} queryElem
         * @param {[FunctionType]} whereClause - Trait bounds for generic items.
         * @param {Map<number,number>|null} mgens - Map functions generics to query generics.
         * @param {number} unboxingDepth
         * @returns {boolean}
         */
        function unifyFunctionTypeIsUnboxCandidate(
            fnType,
            queryElem,
            whereClause,
            mgens,
            unboxingDepth
        ) {
            if (unboxingDepth >= UNBOXING_LIMIT) {
                return false;
            }
            if (fnType.id < 0 && queryElem.id >= 0) {
                if (!whereClause) {
                    return false;
                }
                // mgens[fnType.id] === 0 indicates that we committed to unboxing this generic
                // mgens[fnType.id] === null indicates that we haven't decided yet
                if (mgens && mgens.has(fnType.id) && mgens.get(fnType.id) !== 0) {
                    return false;
                }
                // Where clauses can represent cyclical data.
                // `null` prevents it from trying to unbox in an infinite loop
                const mgensTmp = new Map(mgens);
                mgensTmp.set(fnType.id, null);
                // This is only a potential unbox if the search query appears in the where clause
                // for example, searching `Read -> usize` should find
                // `fn read_all<R: Read>(R) -> Result<usize>`
                // generic `R` is considered "unboxed"
                return checkIfInList(
                    whereClause[(-fnType.id) - 1],
                    queryElem,
                    whereClause,
                    mgensTmp,
                    unboxingDepth
                );
            } else if (fnType.generics.length > 0 || fnType.bindings.size > 0) {
                const simplifiedGenerics = [
                    ...fnType.generics,
                    ...Array.from(fnType.bindings.values()).flat(),
                ];
                return checkIfInList(
                    simplifiedGenerics,
                    queryElem,
                    whereClause,
                    mgens,
                    unboxingDepth
                );
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
          * @param {Map<number,number>|null} mgens - Map functions generics to query generics.
          * @param {number} unboxingDepth
          *
          * @return {boolean} - Returns true if found, false otherwise.
          */
        function checkIfInList(list, elem, whereClause, mgens, unboxingDepth) {
            for (const entry of list) {
                if (checkType(entry, elem, whereClause, mgens, unboxingDepth)) {
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
          * @param {Map<number,number>|null} mgens - Map functions generics to query generics.
          *
          * @return {boolean} - Returns true if the type matches, false otherwise.
          */
        function checkType(row, elem, whereClause, mgens, unboxingDepth) {
            if (unboxingDepth >= UNBOXING_LIMIT) {
                return false;
            }
            if (row.bindings.size === 0 && elem.bindings.size === 0) {
                if (elem.id < 0 && mgens === null) {
                    return row.id < 0 || checkIfInList(
                        row.generics,
                        elem,
                        whereClause,
                        mgens,
                        unboxingDepth + 1
                    );
                }
                if (row.id > 0 && elem.id > 0 && elem.pathWithoutLast.length === 0 &&
                    typePassesFilter(elem.typeFilter, row.ty) && elem.generics.length === 0 &&
                    // special case
                    elem.id !== typeNameIdOfArrayOrSlice && elem.id !== typeNameIdOfTupleOrUnit
                    && elem.id !== typeNameIdOfHof
                ) {
                    return row.id === elem.id || checkIfInList(
                        row.generics,
                        elem,
                        whereClause,
                        mgens,
                        unboxingDepth
                    );
                }
            }
            return unifyFunctionTypes([row], [elem], whereClause, mgens, null, unboxingDepth);
        }

        /**
         * Compute an "edit distance" that ignores missing path elements.
         * @param {string[]} contains search query path
         * @param {Row} ty indexed item
         * @returns {null|number} edit distance
         */
        function checkPath(contains, ty) {
            if (contains.length === 0) {
                return 0;
            }
            const maxPathEditDistance = Math.floor(
                contains.reduce((acc, next) => acc + next.length, 0) / 3
            );
            let ret_dist = maxPathEditDistance + 1;
            const path = ty.path.split("::");

            if (ty.parent && ty.parent.name) {
                path.push(ty.parent.name.toLowerCase());
            }

            const length = path.length;
            const clength = contains.length;
            pathiter: for (let i = length - clength; i >= 0; i -= 1) {
                let dist_total = 0;
                for (let x = 0; x < clength; ++x) {
                    const [p, c] = [path[i + x], contains[x]];
                    if (Math.floor((p.length - c.length) / 3) <= maxPathEditDistance &&
                        p.indexOf(c) !== -1
                    ) {
                        // discount distance on substring match
                        dist_total += Math.floor((p.length - c.length) / 3);
                    } else {
                        const dist = editDistance(p, c, maxPathEditDistance);
                        if (dist > maxPathEditDistance) {
                            continue pathiter;
                        }
                        dist_total += dist;
                    }
                }
                ret_dist = Math.min(ret_dist, Math.round(dist_total / clength));
            }
            return ret_dist > maxPathEditDistance ? null : ret_dist;
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
         * * `id` is the index in the `searchIndex` array for this element.
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
            if (dist <= maxEditDistance || index !== -1) {
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
            let path_dist = 0;
            const fullId = row.id;

            // fpDist is a minimum possible type distance, where "type distance" is the number of
            // atoms in the function not present in the query
            const tfpDist = compareTypeFingerprints(
                fullId,
                parsedQuery.typeFingerprint
            );
            if (tfpDist !== null) {
                const in_args = row.type && row.type.inputs
                    && checkIfInList(row.type.inputs, elem, row.type.where_clause, null, 0);
                const returned = row.type && row.type.output
                    && checkIfInList(row.type.output, elem, row.type.where_clause, null, 0);
                if (in_args) {
                    results_in_args.max_dist = Math.max(results_in_args.max_dist || 0, tfpDist);
                    const maxDist = results_in_args.size < MAX_RESULTS ?
                        (tfpDist + 1) :
                        results_in_args.max_dist;
                    addIntoResults(results_in_args, fullId, pos, -1, tfpDist, 0, maxDist);
                }
                if (returned) {
                    results_returned.max_dist = Math.max(results_returned.max_dist || 0, tfpDist);
                    const maxDist = results_returned.size < MAX_RESULTS ?
                        (tfpDist + 1) :
                        results_returned.max_dist;
                    addIntoResults(results_returned, fullId, pos, -1, tfpDist, 0, maxDist);
                }
            }

            if (!typePassesFilter(elem.typeFilter, row.ty)) {
                return;
            }

            let index = row.word.indexOf(elem.pathLast);
            const normalizedIndex = row.normalizedName.indexOf(elem.pathLast);
            if (index === -1 || (index > normalizedIndex && normalizedIndex !== -1)) {
                index = normalizedIndex;
            }

            if (elem.fullPath.length > 1) {
                path_dist = checkPath(elem.pathWithoutLast, row);
                if (path_dist === null) {
                    return;
                }
            }

            if (parsedQuery.literalSearch) {
                if (row.word === elem.pathLast) {
                    addIntoResults(results_others, fullId, pos, index, 0, path_dist);
                }
                return;
            }

            const dist = editDistance(row.normalizedName, elem.normalizedPathLast, maxEditDistance);

            if (index === -1 && dist > maxEditDistance) {
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

            const tfpDist = compareTypeFingerprints(
                row.id,
                parsedQuery.typeFingerprint
            );
            if (tfpDist === null) {
                return;
            }
            if (results.size >= MAX_RESULTS && tfpDist > results.max_dist) {
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
                        mgens,
                        null,
                        0 // unboxing depth
                    );
                },
                0 // unboxing depth
            )) {
                return;
            }

            results.max_dist = Math.max(results.max_dist || 0, tfpDist);
            addIntoResults(results, row.id, pos, 0, tfpDist, 0, Number.MAX_VALUE);
        }

        function innerRunQuery() {
            const queryLen =
                parsedQuery.elems.reduce((acc, next) => acc + next.pathLast.length, 0) +
                parsedQuery.returned.reduce((acc, next) => acc + next.pathLast.length, 0);
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
             * @param {boolean} isAssocType
             */
            function convertNameToId(elem, isAssocType) {
                if (typeNameIdMap.has(elem.normalizedPathLast) &&
                    (isAssocType || !typeNameIdMap.get(elem.normalizedPathLast).assocOnly)) {
                    elem.id = typeNameIdMap.get(elem.normalizedPathLast).id;
                } else if (!parsedQuery.literalSearch) {
                    let match = null;
                    let matchDist = maxEditDistance + 1;
                    let matchName = "";
                    for (const [name, {id, assocOnly}] of typeNameIdMap) {
                        const dist = editDistance(name, elem.normalizedPathLast, maxEditDistance);
                        if (dist <= matchDist && dist <= maxEditDistance &&
                            (isAssocType || !assocOnly)) {
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
                     && elem.generics.length === 0 && elem.bindings.size === 0)
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
                elem.bindings = new Map(Array.from(elem.bindings.entries())
                    .map(entry => {
                        const [name, constraints] = entry;
                        if (!typeNameIdMap.has(name)) {
                            parsedQuery.error = [
                                "Type parameter ",
                                name,
                                " does not exist",
                            ];
                            return [null, []];
                        }
                        for (const elem2 of constraints) {
                            convertNameToId(elem2);
                        }

                        return [typeNameIdMap.get(name).id, constraints];
                    })
                );
            }

            const fps = new Set();
            for (const elem of parsedQuery.elems) {
                convertNameToId(elem);
                buildFunctionTypeFingerprint(elem, parsedQuery.typeFingerprint, fps);
            }
            for (const elem of parsedQuery.returned) {
                convertNameToId(elem);
                buildFunctionTypeFingerprint(elem, parsedQuery.typeFingerprint, fps);
            }

            if (parsedQuery.foundElems === 1 && parsedQuery.returned.length === 0) {
                if (parsedQuery.elems.length === 1) {
                    const elem = parsedQuery.elems[0];
                    for (let i = 0, nSearchIndex = searchIndex.length; i < nSearchIndex; ++i) {
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
                }
            } else if (parsedQuery.foundElems > 0) {
                // Sort input and output so that generic type variables go first and
                // types with generic parameters go last.
                // That's because of the way unification is structured: it eats off
                // the end, and hits a fast path if the last item is a simple atom.
                const sortQ = (a, b) => {
                    const ag = a.generics.length === 0 && a.bindings.size === 0;
                    const bg = b.generics.length === 0 && b.bindings.size === 0;
                    if (ag !== bg) {
                        return ag - bg;
                    }
                    const ai = a.id > 0;
                    const bi = b.id > 0;
                    return ai - bi;
                };
                parsedQuery.elems.sort(sortQ);
                parsedQuery.returned.sort(sortQ);
                for (let i = 0, nSearchIndex = searchIndex.length; i < nSearchIndex; ++i) {
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
        const extraClass = display ? " active" : "";

        const output = document.createElement("div");
        if (array.length > 0) {
            output.className = "search-results " + extraClass;

            array.forEach(item => {
                const name = item.name;
                const type = itemTypes[item.ty];
                const longType = longItemTypes[item.ty];
                const typeName = longType.length !== 0 ? `${longType}` : "?";

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
        return [output, array.length];
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
        if (rawSearchIndex.size > 1) {
            crates = " in&nbsp;<div id=\"crate-search-div\"><select id=\"crate-search\">" +
                "<option value=\"all crates\">all crates</option>";
            for (const c of rawSearchIndex.keys()) {
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
     * @param {boolean} [forced]
     */
    function search(forced) {
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
            execQuery(query, filterCrates, window.currentCrate),
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
        return types.length > 0 ?
            types.map(type => buildItemSearchType(type, lowercasePaths)) :
            EMPTY_GENERICS_ARRAY;
    }

    /**
     * Empty, immutable map used in item search types with no bindings.
     *
     * @type {Map<number, Array<FunctionType>>}
     */
    const EMPTY_BINDINGS_MAP = new Map();

    /**
     * Empty, immutable map used in item search types with no bindings.
     *
     * @type {Array<FunctionType>}
     */
    const EMPTY_GENERICS_ARRAY = [];

    /**
     * Object pool for function types with no bindings or generics.
     * This is reset after loading the index.
     *
     * @type {Map<number|null, FunctionType>}
     */
    let TYPES_POOL = new Map();

    /**
     * Converts a single type.
     *
     * @param {RawFunctionType} type
     */
    function buildItemSearchType(type, lowercasePaths, isAssocType) {
        const PATH_INDEX_DATA = 0;
        const GENERICS_DATA = 1;
        const BINDINGS_DATA = 2;
        let pathIndex, generics, bindings;
        if (typeof type === "number") {
            pathIndex = type;
            generics = EMPTY_GENERICS_ARRAY;
            bindings = EMPTY_BINDINGS_MAP;
        } else {
            pathIndex = type[PATH_INDEX_DATA];
            generics = buildItemSearchTypeAll(
                type[GENERICS_DATA],
                lowercasePaths
            );
            if (type.length > BINDINGS_DATA && type[BINDINGS_DATA].length > 0) {
                bindings = new Map(type[BINDINGS_DATA].map(binding => {
                    const [assocType, constraints] = binding;
                    // Associated type constructors are represented sloppily in rustdoc's
                    // type search, to make the engine simpler.
                    //
                    // MyType<Output<T>=Result<T>> is equivalent to MyType<Output<Result<T>>=T>
                    // and both are, essentially
                    // MyType<Output=(T, Result<T>)>, except the tuple isn't actually there.
                    // It's more like the value of a type binding is naturally an array,
                    // which rustdoc calls "constraints".
                    //
                    // As a result, the key should never have generics on it.
                    return [
                        buildItemSearchType(assocType, lowercasePaths, true).id,
                        buildItemSearchTypeAll(constraints, lowercasePaths),
                    ];
                }));
            } else {
                bindings = EMPTY_BINDINGS_MAP;
            }
        }
        /**
         * @type {FunctionType}
         */
        let result;
        if (pathIndex < 0) {
            // types less than 0 are generic parameters
            // the actual names of generic parameters aren't stored, since they aren't API
            result = {
                id: pathIndex,
                ty: TY_GENERIC,
                path: null,
                generics,
                bindings,
            };
        } else if (pathIndex === 0) {
            // `0` is used as a sentinel because it's fewer bytes than `null`
            result = {
                id: null,
                ty: null,
                path: null,
                generics,
                bindings,
            };
        } else {
            const item = lowercasePaths[pathIndex - 1];
            result = {
                id: buildTypeMapIndex(item.name, isAssocType),
                ty: item.ty,
                path: item.path,
                generics,
                bindings,
            };
        }
        const cr = TYPES_POOL.get(result.id);
        if (cr) {
            // Shallow equality check. Since this function is used
            // to construct every type object, this should be mostly
            // equivalent to a deep equality check, except if there's
            // a conflict, we don't keep the old one around, so it's
            // not a fully precise implementation of hashcons.
            if (cr.generics.length === result.generics.length &&
                cr.generics !== result.generics &&
                cr.generics.every((x, i) => result.generics[i] === x)
            ) {
                result.generics = cr.generics;
            }
            if (cr.bindings.size === result.bindings.size && cr.bindings !== result.bindings) {
                let ok = true;
                for (const [k, v] of cr.bindings.entries()) {
                    const v2 = result.bindings.get(v);
                    if (!v2) {
                        ok = false;
                        break;
                    }
                    if (v !== v2 && v.length === v2.length && v.every((x, i) => v2[i] === x)) {
                        result.bindings.set(k, v);
                    } else if (v !== v2) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    result.bindings = cr.bindings;
                }
            }
            if (cr.ty === result.ty && cr.path === result.path
                && cr.bindings === result.bindings && cr.generics === result.generics
                && cr.ty === result.ty
            ) {
                return cr;
            }
        }
        TYPES_POOL.set(result.id, result);
        return result;
    }

    /**
     * Convert from RawFunctionSearchType to FunctionSearchType.
     *
     * Crates often have lots of functions in them, and function signatures are sometimes complex,
     * so rustdoc uses a pretty tight encoding for them. This function converts it to a simpler,
     * object-based encoding so that the actual search code is more readable and easier to debug.
     *
     * The raw function search type format is generated using serde in
     * librustdoc/html/render/mod.rs: IndexItemFunctionType::write_to_string
     *
     * @param {{
     *  string: string,
     *  offset: number,
     *  backrefQueue: FunctionSearchType[]
     * }} itemFunctionDecoder
     * @param {Array<{name: string, ty: number}>} lowercasePaths
     * @param {Map<string, integer>}
     *
     * @return {null|FunctionSearchType}
     */
    function buildFunctionSearchType(itemFunctionDecoder, lowercasePaths) {
        const c = itemFunctionDecoder.string.charCodeAt(itemFunctionDecoder.offset);
        itemFunctionDecoder.offset += 1;
        const [zero, ua, la, ob, cb] = ["0", "@", "`", "{", "}"].map(c => c.charCodeAt(0));
        // `` ` `` is used as a sentinel because it's fewer bytes than `null`, and decodes to zero
        // `0` is a backref
        if (c === la) {
            return null;
        }
        // sixteen characters after "0" are backref
        if (c >= zero && c < ua) {
            return itemFunctionDecoder.backrefQueue[c - zero];
        }
        if (c !== ob) {
            throw ["Unexpected ", c, " in function: expected ", "{", "; this is a bug"];
        }
        // call after consuming `{`
        function decodeList() {
            let c = itemFunctionDecoder.string.charCodeAt(itemFunctionDecoder.offset);
            const ret = [];
            while (c !== cb) {
                ret.push(decode());
                c = itemFunctionDecoder.string.charCodeAt(itemFunctionDecoder.offset);
            }
            itemFunctionDecoder.offset += 1; // eat cb
            return ret;
        }
        // consumes and returns a list or integer
        function decode() {
            let n = 0;
            let c = itemFunctionDecoder.string.charCodeAt(itemFunctionDecoder.offset);
            if (c === ob) {
                itemFunctionDecoder.offset += 1;
                return decodeList();
            }
            while (c < la) {
                n = (n << 4) | (c & 0xF);
                itemFunctionDecoder.offset += 1;
                c = itemFunctionDecoder.string.charCodeAt(itemFunctionDecoder.offset);
            }
            // last character >= la
            n = (n << 4) | (c & 0xF);
            const [sign, value] = [n & 1, n >> 1];
            itemFunctionDecoder.offset += 1;
            return sign ? -value : value;
        }
        const functionSearchType = decodeList();
        const INPUTS_DATA = 0;
        const OUTPUT_DATA = 1;
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
        const ret = {
            inputs, output, where_clause,
        };
        itemFunctionDecoder.backrefQueue.unshift(ret);
        if (itemFunctionDecoder.backrefQueue.length > 16) {
            itemFunctionDecoder.backrefQueue.pop();
        }
        return ret;
    }

    /**
     * Type fingerprints allow fast, approximate matching of types.
     *
     * This algo creates a compact representation of the type set using a Bloom filter.
     * This fingerprint is used three ways:
     *
     * - It accelerates the matching algorithm by checking the function fingerprint against the
     *   query fingerprint. If any bits are set in the query but not in the function, it can't
     *   match.
     *
     * - The fourth section has the number of distinct items in the set.
     *   This is the distance function, used for filtering and for sorting.
     *
     * [^1]: Distance is the relatively naive metric of counting the number of distinct items in
     * the function that are not present in the query.
     *
     * @param {FunctionType|QueryElement} type - a single type
     * @param {Uint32Array} output - write the fingerprint to this data structure: uses 128 bits
     * @param {Set<number>} fps - Set of distinct items
     */
    function buildFunctionTypeFingerprint(type, output, fps) {
        let input = type.id;
        // All forms of `[]`/`()`/`->` get collapsed down to one thing in the bloom filter.
        // Differentiating between arrays and slices, if the user asks for it, is
        // still done in the matching algorithm.
        if (input === typeNameIdOfArray || input === typeNameIdOfSlice) {
            input = typeNameIdOfArrayOrSlice;
        }
        if (input === typeNameIdOfTuple || input === typeNameIdOfUnit) {
            input = typeNameIdOfTupleOrUnit;
        }
        if (input === typeNameIdOfFn || input === typeNameIdOfFnMut ||
            input === typeNameIdOfFnOnce) {
            input = typeNameIdOfHof;
        }
        // http://burtleburtle.net/bob/hash/integer.html
        // ~~ is toInt32. It's used before adding, so
        // the number stays in safe integer range.
        const hashint1 = k => {
            k = (~~k + 0x7ed55d16) + (k << 12);
            k = (k ^ 0xc761c23c) ^ (k >>> 19);
            k = (~~k + 0x165667b1) + (k << 5);
            k = (~~k + 0xd3a2646c) ^ (k << 9);
            k = (~~k + 0xfd7046c5) + (k << 3);
            return (k ^ 0xb55a4f09) ^ (k >>> 16);
        };
        const hashint2 = k => {
            k = ~k + (k << 15);
            k ^= k >>> 12;
            k += k << 2;
            k ^= k >>> 4;
            k = Math.imul(k, 2057);
            return k ^ (k >> 16);
        };
        if (input !== null) {
            const h0a = hashint1(input);
            const h0b = hashint2(input);
            // Less Hashing, Same Performance: Building a Better Bloom Filter
            // doi=10.1.1.72.2442
            const h1a = ~~(h0a + Math.imul(h0b, 2));
            const h1b = ~~(h0a + Math.imul(h0b, 3));
            const h2a = ~~(h0a + Math.imul(h0b, 4));
            const h2b = ~~(h0a + Math.imul(h0b, 5));
            output[0] |= (1 << (h0a % 32)) | (1 << (h1b % 32));
            output[1] |= (1 << (h1a % 32)) | (1 << (h2b % 32));
            output[2] |= (1 << (h2a % 32)) | (1 << (h0b % 32));
            fps.add(input);
        }
        for (const g of type.generics) {
            buildFunctionTypeFingerprint(g, output, fps);
        }
        const fb = {
            id: null,
            ty: 0,
            generics: EMPTY_GENERICS_ARRAY,
            bindings: EMPTY_BINDINGS_MAP,
        };
        for (const [k, v] of type.bindings.entries()) {
            fb.id = k;
            fb.generics = v;
            buildFunctionTypeFingerprint(fb, output, fps);
        }
        output[3] = fps.size;
    }

    /**
     * Compare the query fingerprint with the function fingerprint.
     *
     * @param {{number}} fullId - The function
     * @param {{Uint32Array}} queryFingerprint - The query
     * @returns {number|null} - Null if non-match, number if distance
     *                          This function might return 0!
     */
    function compareTypeFingerprints(fullId, queryFingerprint) {
        const fh0 = functionTypeFingerprint[fullId * 4];
        const fh1 = functionTypeFingerprint[(fullId * 4) + 1];
        const fh2 = functionTypeFingerprint[(fullId * 4) + 2];
        const [qh0, qh1, qh2] = queryFingerprint;
        // Approximate set intersection with bloom filters.
        // This can be larger than reality, not smaller, because hashes have
        // the property that if they've got the same value, they hash to the
        // same thing. False positives exist, but not false negatives.
        const [in0, in1, in2] = [fh0 & qh0, fh1 & qh1, fh2 & qh2];
        // Approximate the set of items in the query but not the function.
        // This might be smaller than reality, but cannot be bigger.
        //
        // | in_ | qh_ | XOR | Meaning                                          |
        // | --- | --- | --- | ------------------------------------------------ |
        // |  0  |  0  |  0  | Not present                                      |
        // |  1  |  0  |  1  | IMPOSSIBLE because `in_` is `fh_ & qh_`          |
        // |  1  |  1  |  0  | If one or both is false positive, false negative |
        // |  0  |  1  |  1  | Since in_ has no false negatives, must be real   |
        if ((in0 ^ qh0) || (in1 ^ qh1) || (in2 ^ qh2)) {
            return null;
        }
        return functionTypeFingerprint[(fullId * 4) + 3];
    }

    /**
     * Convert raw search index into in-memory search index.
     *
     * @param {[string, RawSearchIndexCrate][]} rawSearchIndex
     */
    function buildIndex(rawSearchIndex) {
        searchIndex = [];
        const charA = "A".charCodeAt(0);
        let currentIndex = 0;
        let id = 0;

        // Function type fingerprints are 128-bit bloom filters that are used to
        // estimate the distance between function and query.
        // This loop counts the number of items to allocate a fingerprint for.
        for (const crate of rawSearchIndex.values()) {
            // Each item gets an entry in the fingerprint array, and the crate
            // does, too
            id += crate.t.length + 1;
        }
        functionTypeFingerprint = new Uint32Array((id + 1) * 4);

        // This loop actually generates the search item indexes, including
        // normalized names, type signature objects and fingerprints, and aliases.
        id = 0;

        for (const [crate, crateCorpus] of rawSearchIndex) {
            // This object should have exactly the same set of fields as the "row"
            // object defined below. Your JavaScript runtime will thank you.
            // https://mathiasbynens.be/notes/shapes-ics
            const crateRow = {
                crate: crate,
                ty: 3, // == ExternCrate
                name: crate,
                path: "",
                desc: crateCorpus.doc,
                parent: undefined,
                type: null,
                id: id,
                word: crate,
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
            // a string representing the list of function types
            const itemFunctionDecoder = {
                string: crateCorpus.f,
                offset: 0,
                backrefQueue: [],
            };
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
                if (typeof itemNames[i] === "string") {
                    word = itemNames[i].toLowerCase();
                }
                const path = itemPaths.has(i) ? itemPaths.get(i) : lastPath;
                const type = buildFunctionSearchType(itemFunctionDecoder, lowercasePaths);
                if (type !== null) {
                    if (type) {
                        const fp = functionTypeFingerprint.subarray(id * 4, (id + 1) * 4);
                        const fps = new Set();
                        for (const t of type.inputs) {
                            buildFunctionTypeFingerprint(t, fp, fps);
                        }
                        for (const t of type.output) {
                            buildFunctionTypeFingerprint(t, fp, fps);
                        }
                        for (const w of type.where_clause) {
                            for (const t of w) {
                                buildFunctionTypeFingerprint(t, fp, fps);
                            }
                        }
                    }
                }
                // This object should have exactly the same set of fields as the "crateRow"
                // object defined above.
                const row = {
                    crate: crate,
                    ty: itemTypes.charCodeAt(i) - charA,
                    name: itemNames[i],
                    path: path,
                    desc: itemDescs[i],
                    parent: itemParentIdxs[i] > 0 ? paths[itemParentIdxs[i] - 1] : undefined,
                    type,
                    id: id,
                    word,
                    normalizedName: word.indexOf("_") === -1 ? word : word.replace(/_/g, ""),
                    deprecated: deprecatedItems.has(i),
                    implDisambiguator: implDisambiguator.has(i) ? implDisambiguator.get(i) : null,
                };
                id += 1;
                searchIndex.push(row);
                lastPath = row.path;
            }

            if (aliases) {
                const currentCrateAliases = new Map();
                ALIASES.set(crate, currentCrateAliases);
                for (const alias_name in aliases) {
                    if (!Object.prototype.hasOwnProperty.call(aliases, alias_name)) {
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
            currentIndex += itemTypes.length;
        }
        // Drop the (rather large) hash table used for reusing function items
        TYPES_POOL = new Map();
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
                    e.preventDefault();
                    search();
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
        search(true);
    }

    buildIndex(rawSearchIndex);
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
}

if (typeof window !== "undefined") {
    window.initSearch = initSearch;
    if (window.searchIndex !== undefined) {
        initSearch(window.searchIndex);
    }
} else {
    // Running in Node, not a browser. Run initSearch just to produce the
    // exports.
    initSearch(new Map());
}


})();
