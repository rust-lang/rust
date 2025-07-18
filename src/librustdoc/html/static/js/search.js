// ignore-tidy-filelength
/* global addClass, getNakedUrl, getSettingValue, getVar */
/* global onEachLazy, removeClass, searchState, browserSupportsHistoryApi, exports */

"use strict";

// polyfill
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/toSpliced
if (!Array.prototype.toSpliced) {
    // Can't use arrow functions, because we want `this`
    Array.prototype.toSpliced = function() {
        const me = this.slice();
        // @ts-expect-error
        Array.prototype.splice.apply(me, arguments);
        return me;
    };
}

/**
 *
 * @template T
 * @param {Iterable<T>} arr
 * @param {function(T): any} func
 * @param {function(T): boolean} funcBtwn
 */
function onEachBtwn(arr, func, funcBtwn) {
    let skipped = true;
    for (const value of arr) {
        if (!skipped) {
            funcBtwn(value);
        }
        skipped = func(value);
    }
}

/**
 * Convert any `undefined` to `null`.
 *
 * @template T
 * @param {T|undefined} x
 * @returns {T|null}
 */
function undef2null(x) {
    if (x !== undefined) {
        return x;
    }
    return null;
}

// ==================== Core search logic begin ====================
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

// used for special search precedence
const TY_PRIMITIVE = itemTypes.indexOf("primitive");
const TY_GENERIC = itemTypes.indexOf("generic");
const TY_IMPORT = itemTypes.indexOf("import");
const TY_TRAIT = itemTypes.indexOf("trait");
const TY_FN = itemTypes.indexOf("fn");
const TY_METHOD = itemTypes.indexOf("method");
const TY_TYMETHOD = itemTypes.indexOf("tymethod");
const ROOT_PATH = typeof window !== "undefined" ? window.rootPath : "../";

// Hard limit on how deep to recurse into generics when doing type-driven search.
// This needs limited, partially because
// a search for `Ty` shouldn't match `WithInfcx<ParamEnvAnd<Vec<ConstTy<Interner<Ty=Ty>>>>>`,
// but mostly because this is the simplest and most principled way to limit the number
// of permutations we need to check.
const UNBOXING_LIMIT = 5;

// used for search query verification
const REGEX_IDENT = /\p{ID_Start}\p{ID_Continue}*|_\p{ID_Continue}+/uy;
const REGEX_INVALID_TYPE_FILTER = /[^a-z]/ui;

const MAX_RESULTS = 200;
const NO_TYPE_FILTER = -1;

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
    /**
     * @type {number[]}
     */
    current: [],
    /**
     * @type {number[]}
     */
    prev: [],
    /**
     * @type {number[]}
     */
    prevPrev: [],
    /**
     * @param {string} a
     * @param {string} b
     * @param {number} limit
     * @returns
     */
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
                    this.prev[j - 1] + substitutionCost,
                );

                if ((i > 1) && (j > 1) && (a[aIdx] === b[bIdx - 1]) && (a[aIdx - 1] === b[bIdx])) {
                    // transposition
                    this.current[j] = Math.min(
                        this.current[j],
                        this.prevPrev[j - 2] + 1,
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

/**
 * @param {string} a
 * @param {string} b
 * @param {number} limit
 * @returns
 */
function editDistance(a, b, limit) {
    return editDistanceState.calculate(a, b, limit);
}

/**
 * @param {string} c
 * @returns {boolean}
 */
function isEndCharacter(c) {
    return "=,>-])".indexOf(c) !== -1;
}

/**
 * @param {number} ty
 * @returns
 */
function isFnLikeTy(ty) {
    return ty === TY_FN || ty === TY_METHOD || ty === TY_TYMETHOD;
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
 * Returns `true` if the current parser position is starting with "->".
 *
 * @param {rustdoc.ParserState} parserState
 *
 * @return {boolean}
 */
function isReturnArrow(parserState) {
    return parserState.userQuery.slice(parserState.pos, parserState.pos + 2) === "->";
}

/**
 * Increase current parser position until it doesn't find a whitespace anymore.
 *
 * @param {rustdoc.ParserState} parserState
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

/**
 * Returns `true` if the previous character is `lookingFor`.
 *
 * @param {rustdoc.ParserState} parserState
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
 * @param {Array<rustdoc.ParserQueryElement>} elems
 * @param {rustdoc.ParserState} parserState
 *
 * @return {boolean}
 */
function isLastElemGeneric(elems, parserState) {
    return (elems.length > 0 && elems[elems.length - 1].generics.length > 0) ||
        prevIs(parserState, ">");
}

/**
 *
 * @param {rustdoc.ParsedQuery<rustdoc.ParserQueryElement>} query
 * @param {rustdoc.ParserState} parserState
 * @param {rustdoc.ParserQueryElement[]} elems
 * @param {boolean} isInGenerics
 */
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
        // typeFilterElem is not undefined. If it was, the elems.length check would have fired.
        // @ts-expect-error
        parserState.typeFilter = typeFilterElem.normalizedPathLast;
        parserState.pos += 1;
        parserState.totalElems -= 1;
        query.literalSearch = false;
        getNextElem(query, parserState, elems, isInGenerics);
    }
}

/**
 * This function parses the next query element until it finds `endChar`,
 * calling `getNextElem` to collect each element.
 *
 * If there is no `endChar`, this function will implicitly stop at the end
 * without raising an error.
 *
 * @param {rustdoc.ParsedQuery<rustdoc.ParserQueryElement>} query
 * @param {rustdoc.ParserState} parserState
 * @param {Array<rustdoc.ParserQueryElement>} elems
 *     - This is where the new {QueryElement} will be added.
 * @param {string} endChar - This function will stop when it'll encounter this
 *                           character.
 * @returns {{foundSeparator: boolean}}
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
            /** @type {string[]} */
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
        // This case can be encountered if `getNextElem` encountered a "stop character"
        // right from the start. For example if you have `,,` or `<>`. In this case,
        // we simply move up the current position to continue the parsing.
        if (posBefore === parserState.pos) {
            parserState.pos += 1;
        }
        foundStopChar = false;
    }
    if (parserState.pos >= parserState.length && endChar !== "") {
        throw ["Unclosed ", extra];
    }
    // We are either at the end of the string or on the `endChar` character, let's move
    // forward in any case.
    parserState.pos += 1;

    if (hofParameters) {
        // Commas in a HOF don't cause wrapping parens to become a tuple.
        // If you want a one-tuple with a HOF in it, write `((a -> b),)`.
        foundSeparator = false;
        // HOFs can't have directly nested bindings.
        if ([...elems, ...hofParameters].some(x => x.bindingName)
            || parserState.isInBinding) {
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
 * @param {rustdoc.ParsedQuery<rustdoc.ParserQueryElement>} query
 * @param {rustdoc.ParserState} parserState
 * @param {Array<rustdoc.ParserQueryElement>} elems
 *     - This is where the new {QueryElement} will be added.
 * @param {boolean} isInGenerics
 */
function getNextElem(query, parserState, elems, isInGenerics) {
    /** @type {rustdoc.ParserQueryElement[]} */
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
        if (name === "()" && !foundSeparator && generics.length === 1
            && typeFilter === null) {
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
    } else if (parserState.userQuery[parserState.pos] === "&") {
        if (parserState.typeFilter !== null && parserState.typeFilter !== "primitive") {
            throw [
                "Invalid search type: primitive ",
                "&",
                " and ",
                parserState.typeFilter,
                " both specified",
            ];
        }
        parserState.typeFilter = null;
        parserState.pos += 1;
        let c = parserState.userQuery[parserState.pos];
        while (c === " " && parserState.pos < parserState.length) {
            parserState.pos += 1;
            c = parserState.userQuery[parserState.pos];
        }
        const generics = [];
        if (parserState.userQuery.slice(parserState.pos, parserState.pos + 3) === "mut") {
            generics.push(makePrimitiveElement("mut", { typeFilter: "keyword" }));
            parserState.pos += 3;
            c = parserState.userQuery[parserState.pos];
        }
        while (c === " " && parserState.pos < parserState.length) {
            parserState.pos += 1;
            c = parserState.userQuery[parserState.pos];
        }
        if (!isEndCharacter(c) && parserState.pos < parserState.length) {
            getFilteredNextElem(query, parserState, generics, isInGenerics);
        }
        elems.push(makePrimitiveElement("reference", { generics }));
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
                    isInGenerics,
                ),
            );
        }
    }
}

/**
 * Checks that the type filter doesn't have unwanted characters like `<>` (which are ignored
 * if empty).
 *
 * @param {number} start
 * @param {rustdoc.ParserState} parserState
 */
function checkExtraTypeFilterCharacters(start, parserState) {
    const query = parserState.userQuery.slice(start, parserState.pos).trim();

    const match = query.match(REGEX_INVALID_TYPE_FILTER);
    if (match) {
        throw [
            "Unexpected ",
            match[0],
            " in type filter (before ",
            ":",
            ")",
        ];
    }
}

/**
 * @param {rustdoc.ParsedQuery<rustdoc.ParserQueryElement>} query
 * @param {rustdoc.ParserState} parserState
 * @param {string} name - Name of the query element.
 * @param {Array<rustdoc.ParserQueryElement>} generics - List of generics of this query element.
 * @param {boolean} isInGenerics
 *
 * @return {rustdoc.ParserQueryElement} - The newly created `QueryElement`.
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
    if (name.trim() === "!") {
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
    } else if (quadcolon !== null) {
        throw ["Unexpected ", quadcolon[0]];
    }
    const pathSegments = path.split(/(?:::\s*)|(?:\s+(?:::\s*)?)/).map(x => x.toLowerCase());
    // In case we only have something like `<p>`, there is no name.
    if (pathSegments.length === 0
        || (pathSegments.length === 1 && pathSegments[0] === "")) {
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
            if (gen.bindingName !== null && gen.bindingName.name !== null) {
                if (gen.name !== null) {
                    gen.bindingName.generics.unshift(gen);
                }
                bindings.set(
                    gen.bindingName.name.toLowerCase().replace(/_/g, ""),
                    gen.bindingName.generics,
                );
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
 *
 * @param {string|null} name
 * @param {rustdoc.ParserQueryElementFields=} extra
 * @returns {rustdoc.ParserQueryElement}
 */
function makePrimitiveElement(name, extra) {
    return Object.assign({
        name: name,
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
 * If we encounter a `"`, then we try to extract the string
 * from it until we find another `"`.
 *
 * This function will throw an error in the following cases:
 * * There is already another string element.
 * * We are parsing a generic argument.
 * * There is more than one element.
 * * There is no closing `"`.
 *
 * @param {rustdoc.ParsedQuery<rustdoc.ParserQueryElement>} query
 * @param {rustdoc.ParserState} parserState
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
 * This function goes through all characters until it reaches an invalid ident
 * character or the end of the query. It returns the position of the last
 * character of the ident.
 *
 * @param {rustdoc.ParserState} parserState
 *
 * @return {number}
 */
function getIdentEndPosition(parserState) {
    let afterIdent = consumeIdent(parserState);
    let end = parserState.pos;
    let macroExclamation = -1;
    while (parserState.pos < parserState.length) {
        const c = parserState.userQuery[parserState.pos];
        if (c === "!") {
            if (macroExclamation !== -1) {
                throw ["Cannot have more than one ", "!", " in an ident"];
            } else if (parserState.pos + 1 < parserState.length) {
                const pos = parserState.pos;
                parserState.pos++;
                const beforeIdent = consumeIdent(parserState);
                parserState.pos = pos;
                if (beforeIdent) {
                    throw ["Unexpected ", "!", ": it can only be at the end of an ident"];
                }
            }
            if (afterIdent) macroExclamation = parserState.pos;
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
            if (macroExclamation !== -1) {
                throw ["Cannot have associated items in macros"];
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
            throw ["Unexpected ", c, " after ", parserState.userQuery[parserState.pos - 1],
                " (not a valid identifier)"];
        } else {
            throw ["Unexpected ", c, " (not a valid identifier)"];
        }
        parserState.pos += 1;
        afterIdent = consumeIdent(parserState);
        end = parserState.pos;
    }
    if (macroExclamation !== -1) {
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
        end = macroExclamation;
    }
    return end;
}

/**
 * @param {string} c
 * @returns
 */
function isSpecialStartCharacter(c) {
    return "<\"".indexOf(c) !== -1;
}

/**
 * Returns `true` if the current parser position is starting with "::".
 *
 * @param {rustdoc.ParserState} parserState
 *
 * @return {boolean}
 */
function isPathStart(parserState) {
    return parserState.userQuery.slice(parserState.pos, parserState.pos + 2) === "::";
}

/**
 * If the current parser position is at the beginning of an identifier,
 * move the position to the end of it and return `true`. Otherwise, return `false`.
 *
 * @param {rustdoc.ParserState} parserState
 *
 * @return {boolean}
 */
function consumeIdent(parserState) {
    REGEX_IDENT.lastIndex = parserState.pos;
    const match = parserState.userQuery.match(REGEX_IDENT);
    if (match) {
        parserState.pos += match[0].length;
        return true;
    }
    return false;
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
 * @template T
 */
class VlqHexDecoder {
    /**
     * @param {string} string
     * @param {function(rustdoc.VlqData): T} cons
     */
    constructor(string, cons) {
        this.string = string;
        this.cons = cons;
        this.offset = 0;
        /** @type {T[]} */
        this.backrefQueue = [];
    }
    /**
     * call after consuming `{`
     * @returns {rustdoc.VlqData[]}
     */
    decodeList() {
        let c = this.string.charCodeAt(this.offset);
        const ret = [];
        while (c !== 125) { // 125 = "}"
            ret.push(this.decode());
            c = this.string.charCodeAt(this.offset);
        }
        this.offset += 1; // eat cb
        return ret;
    }
    /**
     * consumes and returns a list or integer
     * @returns {rustdoc.VlqData}
     */
    decode() {
        let n = 0;
        let c = this.string.charCodeAt(this.offset);
        if (c === 123) { // 123 = "{"
            this.offset += 1;
            return this.decodeList();
        }
        while (c < 96) { // 96 = "`"
            n = (n << 4) | (c & 0xF);
            this.offset += 1;
            c = this.string.charCodeAt(this.offset);
        }
        // last character >= la
        n = (n << 4) | (c & 0xF);
        const [sign, value] = [n & 1, n >> 1];
        this.offset += 1;
        return sign ? -value : value;
    }
    /**
     * @returns {T}
     */
    next() {
        const c = this.string.charCodeAt(this.offset);
        // sixteen characters after "0" are backref
        if (c >= 48 && c < 64) { // 48 = "0", 64 = "@"
            this.offset += 1;
            return this.backrefQueue[c - 48];
        }
        // special exception: 0 doesn't use backref encoding
        // it's already one character, and it's always nullish
        if (c === 96) { // 96 = "`"
            this.offset += 1;
            return this.cons(0);
        }
        const result = this.cons(this.decode());
        this.backrefQueue.unshift(result);
        if (this.backrefQueue.length > 16) {
            this.backrefQueue.pop();
        }
        return result;
    }
}
class RoaringBitmap {
    /** @param {string} str */
    constructor(str) {
        // https://github.com/RoaringBitmap/RoaringFormatSpec
        //
        // Roaring bitmaps are used for flags that can be kept in their
        // compressed form, even when loaded into memory. This decoder
        // turns the containers into objects, but uses byte array
        // slices of the original format for the data payload.
        const strdecoded = atob(str);
        const u8array = new Uint8Array(strdecoded.length);
        for (let j = 0; j < strdecoded.length; ++j) {
            u8array[j] = strdecoded.charCodeAt(j);
        }
        const has_runs = u8array[0] === 0x3b;
        const size = has_runs ?
            ((u8array[2] | (u8array[3] << 8)) + 1) :
            ((u8array[4] | (u8array[5] << 8) | (u8array[6] << 16) | (u8array[7] << 24)));
        let i = has_runs ? 4 : 8;
        let is_run;
        if (has_runs) {
            const is_run_len = Math.floor((size + 7) / 8);
            is_run = u8array.slice(i, i + is_run_len);
            i += is_run_len;
        } else {
            is_run = new Uint8Array();
        }
        this.keys = [];
        this.cardinalities = [];
        for (let j = 0; j < size; ++j) {
            this.keys.push(u8array[i] | (u8array[i + 1] << 8));
            i += 2;
            this.cardinalities.push((u8array[i] | (u8array[i + 1] << 8)) + 1);
            i += 2;
        }
        this.containers = [];
        let offsets = null;
        if (!has_runs || this.keys.length >= 4) {
            offsets = [];
            for (let j = 0; j < size; ++j) {
                offsets.push(u8array[i] | (u8array[i + 1] << 8) | (u8array[i + 2] << 16) |
                    (u8array[i + 3] << 24));
                i += 4;
            }
        }
        for (let j = 0; j < size; ++j) {
            if (offsets && offsets[j] !== i) {
                // eslint-disable-next-line no-console
                console.log(this.containers);
                throw new Error(`corrupt bitmap ${j}: ${i} / ${offsets[j]}`);
            }
            if (is_run[j >> 3] & (1 << (j & 0x7))) {
                const runcount = (u8array[i] | (u8array[i + 1] << 8));
                i += 2;
                this.containers.push(new RoaringBitmapRun(
                    runcount,
                    u8array.slice(i, i + (runcount * 4)),
                ));
                i += runcount * 4;
            } else if (this.cardinalities[j] >= 4096) {
                this.containers.push(new RoaringBitmapBits(u8array.slice(i, i + 8192)));
                i += 8192;
            } else {
                const end = this.cardinalities[j] * 2;
                this.containers.push(new RoaringBitmapArray(
                    this.cardinalities[j],
                    u8array.slice(i, i + end),
                ));
                i += end;
            }
        }
    }
    /** @param {number} keyvalue */
    contains(keyvalue) {
        const key = keyvalue >> 16;
        const value = keyvalue & 0xFFFF;
        // Binary search algorithm copied from
        // https://en.wikipedia.org/wiki/Binary_search#Procedure
        //
        // Format is required by specification to be sorted.
        // Because keys are 16 bits and unique, length can't be
        // bigger than 2**16, and because we have 32 bits of safe int,
        // left + right can't overflow.
        let left = 0;
        let right = this.keys.length - 1;
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            const x = this.keys[mid];
            if (x < key) {
                left = mid + 1;
            } else if (x > key) {
                right = mid - 1;
            } else {
                return this.containers[mid].contains(value);
            }
        }
        return false;
    }
}

class RoaringBitmapRun {
    /**
     * @param {number} runcount
     * @param {Uint8Array} array
     */
    constructor(runcount, array) {
        this.runcount = runcount;
        this.array = array;
    }
    /** @param {number} value */
    contains(value) {
        // Binary search algorithm copied from
        // https://en.wikipedia.org/wiki/Binary_search#Procedure
        //
        // Since runcount is stored as 16 bits, left + right
        // can't overflow.
        let left = 0;
        let right = this.runcount - 1;
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            const i = mid * 4;
            const start = this.array[i] | (this.array[i + 1] << 8);
            const lenm1 = this.array[i + 2] | (this.array[i + 3] << 8);
            if ((start + lenm1) < value) {
                left = mid + 1;
            } else if (start > value) {
                right = mid - 1;
            } else {
                return true;
            }
        }
        return false;
    }
}
class RoaringBitmapArray {
    /**
     * @param {number} cardinality
     * @param {Uint8Array} array
     */
    constructor(cardinality, array) {
        this.cardinality = cardinality;
        this.array = array;
    }
    /** @param {number} value */
    contains(value) {
        // Binary search algorithm copied from
        // https://en.wikipedia.org/wiki/Binary_search#Procedure
        //
        // Since cardinality can't be higher than 4096, left + right
        // cannot overflow.
        let left = 0;
        let right = this.cardinality - 1;
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            const i = mid * 2;
            const x = this.array[i] | (this.array[i + 1] << 8);
            if (x < value) {
                left = mid + 1;
            } else if (x > value) {
                right = mid - 1;
            } else {
                return true;
            }
        }
        return false;
    }
}
class RoaringBitmapBits {
    /**
     * @param {Uint8Array} array
     */
    constructor(array) {
        this.array = array;
    }
    /** @param {number} value */
    contains(value) {
        return !!(this.array[value >> 3] & (1 << (value & 7)));
    }
}

/**
 * A prefix tree, used for name-based search.
 *
 * This data structure is used to drive prefix matches,
 * such as matching the query "link" to `LinkedList`,
 * and Lev-distance matches, such as matching the
 * query "hahsmap" to `HashMap`. Substring matches,
 * such as "list" to `LinkedList`, are done with a
 * tailTable that deep-links into this trie.
 *
 * children
 * : A [sparse array] of subtrees. The array index
 *   is a charCode.
 *
 *   [sparse array]: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/
 *     Indexed_collections#sparse_arrays
 *
 * matches
 * : A list of search index IDs for this node.
 *
 * @type {{
 *     children: NameTrie[],
 *     matches: number[],
 * }}
 */
class NameTrie {
    constructor() {
        this.children = [];
        this.matches = [];
    }
    /**
     * @param {string} name
     * @param {number} id
     * @param {Map<string, NameTrie[]>} tailTable
     */
    insert(name, id, tailTable) {
        this.insertSubstring(name, 0, id, tailTable);
    }
    /**
     * @param {string} name
     * @param {number} substart
     * @param {number} id
     * @param {Map<string, NameTrie[]>} tailTable
     */
    insertSubstring(name, substart, id, tailTable) {
        const l = name.length;
        if (substart === l) {
            this.matches.push(id);
        } else {
            const sb = name.charCodeAt(substart);
            let child;
            if (this.children[sb] !== undefined) {
                child = this.children[sb];
            } else {
                child = new NameTrie();
                this.children[sb] = child;
                /** @type {NameTrie[]} */
                let sste;
                if (substart >= 2) {
                    const tail = name.substring(substart - 2, substart + 1);
                    const entry = tailTable.get(tail);
                    if (entry !== undefined) {
                        sste = entry;
                    } else {
                        sste = [];
                        tailTable.set(tail, sste);
                    }
                    sste.push(child);
                }
            }
            child.insertSubstring(name, substart + 1, id, tailTable);
        }
    }
    /**
     * @param {string} name
     * @param {Map<string, NameTrie[]>} tailTable
     */
    search(name, tailTable) {
        const results = new Set();
        this.searchSubstringPrefix(name, 0, results);
        if (results.size < MAX_RESULTS && name.length >= 3) {
            const levParams = name.length >= 6 ?
                new Lev2TParametricDescription(name.length) :
                new Lev1TParametricDescription(name.length);
            this.searchLev(name, 0, levParams, results);
            const tail = name.substring(0, 3);
            const list = tailTable.get(tail);
            if (list !== undefined) {
                for (const entry of list) {
                    entry.searchSubstringPrefix(name, 3, results);
                }
            }
        }
        return [...results];
    }
    /**
     * @param {string} name
     * @param {number} substart
     * @param {Set<number>} results
     */
    searchSubstringPrefix(name, substart, results) {
        const l = name.length;
        if (substart === l) {
            for (const match of this.matches) {
                results.add(match);
            }
            // breadth-first traversal orders prefix matches by length
            /** @type {NameTrie[]} */
            let unprocessedChildren = [];
            for (const child of this.children) {
                if (child) {
                    unprocessedChildren.push(child);
                }
            }
            /** @type {NameTrie[]} */
            let nextSet = [];
            while (unprocessedChildren.length !== 0) {
                /** @type {NameTrie} */
                // @ts-expect-error
                const next = unprocessedChildren.pop();
                for (const child of next.children) {
                    if (child) {
                        nextSet.push(child);
                    }
                }
                for (const match of next.matches) {
                    results.add(match);
                }
                if (unprocessedChildren.length === 0) {
                    const tmp = unprocessedChildren;
                    unprocessedChildren = nextSet;
                    nextSet = tmp;
                }
            }
        } else {
            const sb = name.charCodeAt(substart);
            if (this.children[sb] !== undefined) {
                this.children[sb].searchSubstringPrefix(name, substart + 1, results);
            }
        }
    }
    /**
     * @param {string} name
     * @param {number} substart
     * @param {Lev2TParametricDescription|Lev1TParametricDescription} levParams
     * @param {Set<number>} results
     */
    searchLev(name, substart, levParams, results) {
        const stack = [[this, 0]];
        const n = levParams.n;
        while (stack.length !== 0) {
            // It's not empty
            //@ts-expect-error
            const [trie, levState] = stack.pop();
            for (const [charCode, child] of trie.children.entries()) {
                if (!child) {
                    continue;
                }
                const levPos = levParams.getPosition(levState);
                const vector = levParams.getVector(
                    name,
                    charCode,
                    levPos,
                    Math.min(name.length, levPos + (2 * n) + 1),
                );
                const newLevState = levParams.transition(
                    levState,
                    levPos,
                    vector,
                );
                if (newLevState >= 0) {
                    stack.push([child, newLevState]);
                    if (levParams.isAccept(newLevState)) {
                        for (const match of child.matches) {
                            results.add(match);
                        }
                    }
                }
            }
        }
    }
}

class DocSearch {
    /**
     * @param {Map<string, rustdoc.RawSearchIndexCrate>} rawSearchIndex
     * @param {string} rootPath
     * @param {rustdoc.SearchState} searchState
     */
    constructor(rawSearchIndex, rootPath, searchState) {
        /**
         * @type {Map<String, RoaringBitmap>}
         */
        this.searchIndexDeprecated = new Map();
        /**
         * @type {Map<String, RoaringBitmap>}
         */
        this.searchIndexEmptyDesc = new Map();
        /**
         *  @type {Uint32Array}
         */
        this.functionTypeFingerprint = new Uint32Array(0);
        /**
         * Map from normalized type names to integers. Used to make type search
         * more efficient.
         *
         * @type {Map<string, {id: number, assocOnly: boolean}>}
         */
        this.typeNameIdMap = new Map();
        /**
         * Map from type ID to associated type name. Used for display,
         * not for search.
         *
         * @type {Map<number, string>}
         */
        this.assocTypeIdNameMap = new Map();
        this.ALIASES = new Map();
        this.rootPath = rootPath;
        this.searchState = searchState;

        /**
         * Special type name IDs for searching by array.
         * @type {number}
         */
        this.typeNameIdOfArray = this.buildTypeMapIndex("array");
        /**
         * Special type name IDs for searching by slice.
         * @type {number}
         */
        this.typeNameIdOfSlice = this.buildTypeMapIndex("slice");
        /**
         * Special type name IDs for searching by both array and slice (`[]` syntax).
         * @type {number}
         */
        this.typeNameIdOfArrayOrSlice = this.buildTypeMapIndex("[]");
        /**
         * Special type name IDs for searching by tuple.
         * @type {number}
         */
        this.typeNameIdOfTuple = this.buildTypeMapIndex("tuple");
        /**
         * Special type name IDs for searching by unit.
         * @type {number}
         */
        this.typeNameIdOfUnit = this.buildTypeMapIndex("unit");
        /**
         * Special type name IDs for searching by both tuple and unit (`()` syntax).
         * @type {number}
         */
        this.typeNameIdOfTupleOrUnit = this.buildTypeMapIndex("()");
        /**
         * Special type name IDs for searching `fn`.
         * @type {number}
         */
        this.typeNameIdOfFn = this.buildTypeMapIndex("fn");
        /**
         * Special type name IDs for searching `fnmut`.
         * @type {number}
         */
        this.typeNameIdOfFnMut = this.buildTypeMapIndex("fnmut");
        /**
         * Special type name IDs for searching `fnonce`.
         * @type {number}
         */
        this.typeNameIdOfFnOnce = this.buildTypeMapIndex("fnonce");
        /**
         * Special type name IDs for searching higher order functions (`->` syntax).
         * @type {number}
         */
        this.typeNameIdOfHof = this.buildTypeMapIndex("->");
        /**
         * Special type name IDs the output assoc type.
         * @type {number}
         */
        this.typeNameIdOfOutput = this.buildTypeMapIndex("output", true);
        /**
         * Special type name IDs for searching by reference.
         * @type {number}
         */
        this.typeNameIdOfReference = this.buildTypeMapIndex("reference");

        /**
         * Empty, immutable map used in item search types with no bindings.
         *
         * @type {Map<number, Array<any>>}
         */
        this.EMPTY_BINDINGS_MAP = new Map();

        /**
         * Empty, immutable map used in item search types with no bindings.
         *
         * @type {Array<any>}
         */
        this.EMPTY_GENERICS_ARRAY = [];

        /**
         * Object pool for function types with no bindings or generics.
         * This is reset after loading the index.
         *
         * @type {Map<number|null, rustdoc.FunctionType>}
         */
        this.TYPES_POOL = new Map();

        /**
         * A trie for finding items by name.
         * This is used for edit distance and prefix finding.
         *
         * @type {NameTrie}
         */
        this.nameTrie = new NameTrie();

        /**
         * Find items by 3-substring. This is a map from three-char
         * prefixes into lists of subtries.
         */
        this.tailTable = new Map();

        /**
         *  @type {Array<rustdoc.Row>}
         */
        this.searchIndex = this.buildIndex(rawSearchIndex);
    }

    /**
     * Add an item to the type Name->ID map, or, if one already exists, use it.
     * Returns the number. If name is "" or null, return null (pure generic).
     *
     * This is effectively string interning, so that function matching can be
     * done more quickly. Two types with the same name but different item kinds
     * get the same ID.
     *
     * @template T extends string
     * @overload
     * @param {T} name
     * @param {boolean=} isAssocType - True if this is an assoc type
     * @returns {T extends "" ? null : number}
     *
     * @param {string} name
     * @param {boolean=} isAssocType
     * @returns {number | null}
     *
     */
    buildTypeMapIndex(name, isAssocType) {
        if (name === "" || name === null) {
            return null;
        }

        const obj = this.typeNameIdMap.get(name);
        if (obj !== undefined) {
            obj.assocOnly = !!(isAssocType && obj.assocOnly);
            return obj.id;
        } else {
            const id = this.typeNameIdMap.size;
            this.typeNameIdMap.set(name, { id, assocOnly: !!isAssocType });
            return id;
        }
    }

    /**
     * Convert a list of RawFunctionType / ID to object-based FunctionType.
     *
     * Crates often have lots of functions in them, and it's common to have a large number of
     * functions that operate on a small set of data types, so the search index compresses them
     * by encoding function parameter and return types as indexes into an array of names.
     *
     * Even when a general-purpose compression algorithm is used, this is still a win.
     * I checked. https://github.com/rust-lang/rust/pull/98475#issue-1284395985
     *
     * The format for individual function types is encoded in
     * librustdoc/html/render/mod.rs: impl Serialize for RenderType
     *
     * @param {null|Array<rustdoc.RawFunctionType>} types
     * @param {Array<{
     *     name: string,
    *     ty: number,
    *     path: string|null,
    *     exactPath: string|null,
    *     unboxFlag: boolean
    * }>} paths
    * @param {Array<{
    *     name: string,
    *     ty: number,
    *     path: string|null,
    *     exactPath: string|null,
    *     unboxFlag: boolean,
    * }>} lowercasePaths
     *
     * @return {Array<rustdoc.FunctionType>}
     */
    buildItemSearchTypeAll(types, paths, lowercasePaths) {
        return types && types.length > 0 ?
            types.map(type => this.buildItemSearchType(type, paths, lowercasePaths)) :
            this.EMPTY_GENERICS_ARRAY;
    }

    /**
     * Converts a single type.
     *
     * @param {rustdoc.RawFunctionType} type
     * @param {Array<{
     *     name: string,
     *     ty: number,
     *     path: string|null,
     *     exactPath: string|null,
     *     unboxFlag: boolean
     * }>} paths
     * @param {Array<{
     *     name: string,
     *     ty: number,
     *     path: string|null,
     *     exactPath: string|null,
     *     unboxFlag: boolean,
     * }>} lowercasePaths
     * @param {boolean=} isAssocType
     */
    buildItemSearchType(type, paths, lowercasePaths, isAssocType) {
        const PATH_INDEX_DATA = 0;
        const GENERICS_DATA = 1;
        const BINDINGS_DATA = 2;
        let pathIndex, generics, bindings;
        if (typeof type === "number") {
            pathIndex = type;
            generics = this.EMPTY_GENERICS_ARRAY;
            bindings = this.EMPTY_BINDINGS_MAP;
        } else {
            pathIndex = type[PATH_INDEX_DATA];
            generics = this.buildItemSearchTypeAll(
                type[GENERICS_DATA],
                paths,
                lowercasePaths,
            );
            // @ts-expect-error
            if (type.length > BINDINGS_DATA && type[BINDINGS_DATA].length > 0) {
                // @ts-expect-error
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
                        this.buildItemSearchType(assocType, paths, lowercasePaths, true).id,
                        this.buildItemSearchTypeAll(constraints, paths, lowercasePaths),
                    ];
                }));
            } else {
                bindings = this.EMPTY_BINDINGS_MAP;
            }
        }
        /**
         * @type {rustdoc.FunctionType}
         */
        let result;
        if (pathIndex < 0) {
            // types less than 0 are generic parameters
            // the actual names of generic parameters aren't stored, since they aren't API
            result = {
                id: pathIndex,
                name: "",
                ty: TY_GENERIC,
                path: null,
                exactPath: null,
                generics,
                bindings,
                unboxFlag: true,
            };
        } else if (pathIndex === 0) {
            // `0` is used as a sentinel because it's fewer bytes than `null`
            result = {
                id: null,
                name: "",
                ty: null,
                path: null,
                exactPath: null,
                generics,
                bindings,
                unboxFlag: true,
            };
        } else {
            const item = lowercasePaths[pathIndex - 1];
            const id = this.buildTypeMapIndex(item.name, isAssocType);
            if (isAssocType && id !== null) {
                this.assocTypeIdNameMap.set(id, paths[pathIndex - 1].name);
            }
            result = {
                id,
                name: paths[pathIndex - 1].name,
                ty: item.ty,
                path: item.path,
                exactPath: item.exactPath,
                generics,
                bindings,
                unboxFlag: item.unboxFlag,
            };
        }
        const cr = this.TYPES_POOL.get(result.id);
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
                    // @ts-expect-error
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
                && cr.ty === result.ty && cr.name === result.name
                && cr.unboxFlag === result.unboxFlag
            ) {
                return cr;
            }
        }
        this.TYPES_POOL.set(result.id, result);
        return result;
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
     * - The fourth section has the number of items in the set.
     *   This is the distance function, used for filtering and for sorting.
     *
     * [^1]: Distance is the relatively naive metric of counting the number of distinct items in
     * the function that are not present in the query.
     *
     * @param {rustdoc.FingerprintableType} type - a single type
     * @param {Uint32Array} output - write the fingerprint to this data structure: uses 128 bits
     */
    buildFunctionTypeFingerprint(type, output) {
        let input = type.id;
        // All forms of `[]`/`()`/`->` get collapsed down to one thing in the bloom filter.
        // Differentiating between arrays and slices, if the user asks for it, is
        // still done in the matching algorithm.
        if (input === this.typeNameIdOfArray || input === this.typeNameIdOfSlice) {
            input = this.typeNameIdOfArrayOrSlice;
        }
        if (input === this.typeNameIdOfTuple || input === this.typeNameIdOfUnit) {
            input = this.typeNameIdOfTupleOrUnit;
        }
        if (input === this.typeNameIdOfFn || input === this.typeNameIdOfFnMut ||
            input === this.typeNameIdOfFnOnce) {
            input = this.typeNameIdOfHof;
        }
        /**
         * http://burtleburtle.net/bob/hash/integer.html
         * ~~ is toInt32. It's used before adding, so
         * the number stays in safe integer range.
         * @param {number} k
         */
        const hashint1 = k => {
            k = (~~k + 0x7ed55d16) + (k << 12);
            k = (k ^ 0xc761c23c) ^ (k >>> 19);
            k = (~~k + 0x165667b1) + (k << 5);
            k = (~~k + 0xd3a2646c) ^ (k << 9);
            k = (~~k + 0xfd7046c5) + (k << 3);
            return (k ^ 0xb55a4f09) ^ (k >>> 16);
        };
        /** @param {number} k */
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
            // output[3] is the total number of items in the type signature
            output[3] += 1;
        }
        for (const g of type.generics) {
            this.buildFunctionTypeFingerprint(g, output);
        }
        /**
         * @type {{
         *   id: number|null,
         *   ty: number,
         *   generics: rustdoc.FingerprintableType[],
         *   bindings: Map<number, rustdoc.FingerprintableType[]>
         * }}
         */
        const fb = {
            id: null,
            ty: 0,
            generics: this.EMPTY_GENERICS_ARRAY,
            bindings: this.EMPTY_BINDINGS_MAP,
        };
        for (const [k, v] of type.bindings.entries()) {
            fb.id = k;
            fb.generics = v;
            this.buildFunctionTypeFingerprint(fb, output);
        }
    }

    /**
     * Convert raw search index into in-memory search index.
     *
     * @param {Map<string, rustdoc.RawSearchIndexCrate>} rawSearchIndex
     * @returns {rustdoc.Row[]}
     */
    buildIndex(rawSearchIndex) {
        /**
         * Convert from RawFunctionSearchType to FunctionSearchType.
         *
         * Crates often have lots of functions in them, and function signatures are sometimes
         * complex, so rustdoc uses a pretty tight encoding for them. This function converts it
         * to a simpler, object-based encoding so that the actual search code is more readable
         * and easier to debug.
         *
         * The raw function search type format is generated using serde in
         * librustdoc/html/render/mod.rs: IndexItemFunctionType::write_to_string
         *
         * @param {Array<{
         *     name: string,
         *     ty: number,
         *     path: string|null,
         *     exactPath: string|null,
         *     unboxFlag: boolean
         * }>} paths
         * @param {Array<{
         *     name: string,
         *     ty: number,
         *     path: string|null,
         *     exactPath: string|null,
         *     unboxFlag: boolean
         * }>} lowercasePaths
         *
         * @return {function(rustdoc.RawFunctionSearchType): null|rustdoc.FunctionSearchType}
         */
        const buildFunctionSearchTypeCallback = (paths, lowercasePaths) => {
            /**
             * @param {rustdoc.RawFunctionSearchType} functionSearchType
             */
            const cb = functionSearchType => {
                if (functionSearchType === 0) {
                    return null;
                }
                const INPUTS_DATA = 0;
                const OUTPUT_DATA = 1;
                /** @type {rustdoc.FunctionType[]} */
                let inputs;
                /** @type {rustdoc.FunctionType[]} */
                let output;
                if (typeof functionSearchType[INPUTS_DATA] === "number") {
                    inputs = [
                        this.buildItemSearchType(
                            functionSearchType[INPUTS_DATA],
                            paths,
                            lowercasePaths,
                        ),
                    ];
                } else {
                    inputs = this.buildItemSearchTypeAll(
                        functionSearchType[INPUTS_DATA],
                        paths,
                        lowercasePaths,
                    );
                }
                if (functionSearchType.length > 1) {
                    if (typeof functionSearchType[OUTPUT_DATA] === "number") {
                        output = [
                            this.buildItemSearchType(
                                functionSearchType[OUTPUT_DATA],
                                paths,
                                lowercasePaths,
                            ),
                        ];
                    } else {
                        output = this.buildItemSearchTypeAll(
                            // @ts-expect-error
                            functionSearchType[OUTPUT_DATA],
                            paths,
                            lowercasePaths,
                        );
                    }
                } else {
                    output = [];
                }
                const where_clause = [];
                const l = functionSearchType.length;
                for (let i = 2; i < l; ++i) {
                    where_clause.push(typeof functionSearchType[i] === "number"
                        // @ts-expect-error
                        ? [this.buildItemSearchType(functionSearchType[i], paths, lowercasePaths)]
                        : this.buildItemSearchTypeAll(
                            // @ts-expect-error
                            functionSearchType[i],
                            paths,
                            lowercasePaths,
                        ));
                }
                return {
                    inputs, output, where_clause,
                };
            };
            return cb;
        };

        /** @type {rustdoc.Row[]} */
        const searchIndex = [];
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
        this.functionTypeFingerprint = new Uint32Array((id + 1) * 4);
        // This loop actually generates the search item indexes, including
        // normalized names, type signature objects and fingerprints, and aliases.
        id = 0;

        for (const [crate, crateCorpus] of rawSearchIndex) {
            // a string representing the lengths of each description shard
            // a string representing the list of function types
            const itemDescShardDecoder = new VlqHexDecoder(crateCorpus.D, noop => {
                /** @type {number} */
                // @ts-expect-error
                const n = noop;
                return n;
            });
            let descShard = {
                crate,
                shard: 0,
                start: 0,
                len: itemDescShardDecoder.next(),
                promise: null,
                resolve: null,
            };
            const descShardList = [descShard];

            // Deprecated items and items with no description
            this.searchIndexDeprecated.set(crate, new RoaringBitmap(crateCorpus.c));
            this.searchIndexEmptyDesc.set(crate, new RoaringBitmap(crateCorpus.e));
            let descIndex = 0;

            /**
             * List of generic function type parameter names.
             * Used for display, not for searching.
             * @type {string[]}
             */
            let lastParamNames = [];

            // This object should have exactly the same set of fields as the "row"
            // object defined below. Your JavaScript runtime will thank you.
            // https://mathiasbynens.be/notes/shapes-ics
            let normalizedName = crate.indexOf("_") === -1 ? crate : crate.replace(/_/g, "");
            const crateRow = {
                crate,
                ty: 3, // == ExternCrate
                name: crate,
                path: "",
                descShard,
                descIndex,
                exactPath: "",
                desc: crateCorpus.doc,
                parent: undefined,
                type: null,
                paramNames: lastParamNames,
                id,
                word: crate,
                normalizedName,
                bitIndex: 0,
                implDisambiguator: null,
            };
            this.nameTrie.insert(normalizedName, id, this.tailTable);
            id += 1;
            searchIndex.push(crateRow);
            currentIndex += 1;
            // it's not undefined
            // @ts-expect-error
            if (!this.searchIndexEmptyDesc.get(crate).contains(0)) {
                descIndex += 1;
            }

            // see `RawSearchIndexCrate` in `rustdoc.d.ts` for a more
            // up to date description of these fields
            const itemTypes = crateCorpus.t;
            // an array of (String) item names
            const itemNames = crateCorpus.n;
            // an array of [(Number) item index,
            //              (String) full path]
            // an item whose index is not present will fall back to the previous present path
            // i.e. if indices 4 and 11 are present, but 5-10 and 12-13 are not present,
            // 5-10 will fall back to the path for 4 and 12-13 will fall back to the path for 11
            const itemPaths = new Map(crateCorpus.q);
            // An array of [(Number) item index, (Number) path index]
            // Used to de-duplicate inlined and re-exported stuff
            const itemReexports = new Map(crateCorpus.r);
            // an array of (Number) the parent path index + 1 to `paths`, or 0 if none
            const itemParentIdxDecoder = new VlqHexDecoder(crateCorpus.i, noop => noop);
            // a map Number, string for impl disambiguators
            const implDisambiguator = new Map(crateCorpus.b);
            const rawPaths = crateCorpus.p;
            const aliases = crateCorpus.a;
            // an array of [(Number) item index,
            //              (String) comma-separated list of function generic param names]
            // an item whose index is not present will fall back to the previous present path
            const itemParamNames = new Map(crateCorpus.P);

            /**
             * @type {Array<{
             *     name: string,
             *     ty: number,
             *     path: string|null,
             *     exactPath: string|null,
             *     unboxFlag: boolean
             * }>}
             */
            const lowercasePaths = [];
            /**
             * @type {Array<{
             *     name: string,
             *     ty: number,
             *     path: string|null,
             *     exactPath: string|null,
             *     unboxFlag: boolean
             * }>}
             */
            const paths = [];

            // a string representing the list of function types
            const itemFunctionDecoder = new VlqHexDecoder(
                crateCorpus.f,
                // @ts-expect-error
                buildFunctionSearchTypeCallback(paths, lowercasePaths),
            );

            // convert `rawPaths` entries into object form
            // generate normalizedPaths for function search mode
            let len = rawPaths.length;
            let lastPath = undef2null(itemPaths.get(0));
            for (let i = 0; i < len; ++i) {
                const elem = rawPaths[i];
                const ty = elem[0];
                const name = elem[1];
                /**
                 * @param {2|3} idx
                 * @param {string|null} if_null
                 * @param {string|null} if_not_found
                 * @returns {string|null}
                 */
                const elemPath = (idx, if_null, if_not_found) => {
                    if (elem.length > idx && elem[idx] !== undefined) {
                        const p = itemPaths.get(elem[idx]);
                        if (p !== undefined) {
                            return p;
                        }
                        return if_not_found;
                    }
                    return if_null;
                };
                const path = elemPath(2, lastPath, null);
                const exactPath = elemPath(3, path, path);
                const unboxFlag = elem.length > 4 && !!elem[4];

                lowercasePaths.push({ ty, name: name.toLowerCase(), path, exactPath, unboxFlag });
                paths[i] = { ty, name, path, exactPath, unboxFlag };
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
            let lastName = "";
            let lastWord = "";
            for (let i = 0; i < len; ++i) {
                const bitIndex = i + 1;
                if (descIndex >= descShard.len &&
                    // @ts-expect-error
                    !this.searchIndexEmptyDesc.get(crate).contains(bitIndex)) {
                    descShard = {
                        crate,
                        shard: descShard.shard + 1,
                        start: descShard.start + descShard.len,
                        len: itemDescShardDecoder.next(),
                        promise: null,
                        resolve: null,
                    };
                    descIndex = 0;
                    descShardList.push(descShard);
                }
                const name = itemNames[i] === "" ? lastName : itemNames[i];
                const word = itemNames[i] === "" ? lastWord : itemNames[i].toLowerCase();
                const pathU = itemPaths.get(i);
                const path = pathU !== undefined ? pathU : lastPath;
                const paramNameString = itemParamNames.get(i);
                const paramNames = paramNameString !== undefined ?
                    paramNameString.split(",") :
                    lastParamNames;
                const type = itemFunctionDecoder.next();
                if (type !== null) {
                    if (type) {
                        const fp = this.functionTypeFingerprint.subarray(id * 4, (id + 1) * 4);
                        for (const t of type.inputs) {
                            this.buildFunctionTypeFingerprint(t, fp);
                        }
                        for (const t of type.output) {
                            this.buildFunctionTypeFingerprint(t, fp);
                        }
                        for (const w of type.where_clause) {
                            for (const t of w) {
                                this.buildFunctionTypeFingerprint(t, fp);
                            }
                        }
                    }
                }
                // This object should have exactly the same set of fields as the "crateRow"
                // object defined above.
                const itemParentIdx = itemParentIdxDecoder.next();
                normalizedName = word.indexOf("_") === -1 ? word : word.replace(/_/g, "");
                /** @type {rustdoc.Row} */
                const row = {
                    crate,
                    ty: itemTypes.charCodeAt(i) - 65, // 65 = "A"
                    name,
                    path,
                    descShard,
                    descIndex,
                    exactPath: itemReexports.has(i) ?
                        // @ts-expect-error
                        itemPaths.get(itemReexports.get(i)) : path,
                    // @ts-expect-error
                    parent: itemParentIdx > 0 ? paths[itemParentIdx - 1] : undefined,
                    type,
                    paramNames,
                    id,
                    word,
                    normalizedName,
                    bitIndex,
                    implDisambiguator: undef2null(implDisambiguator.get(i)),
                };
                this.nameTrie.insert(normalizedName, id, this.tailTable);
                id += 1;
                searchIndex.push(row);
                lastPath = row.path;
                lastParamNames = row.paramNames;
                // @ts-expect-error
                if (!this.searchIndexEmptyDesc.get(crate).contains(bitIndex)) {
                    descIndex += 1;
                }
                lastName = name;
                lastWord = word;
            }

            if (aliases) {
                const currentCrateAliases = new Map();
                this.ALIASES.set(crate, currentCrateAliases);
                for (const alias_name in aliases) {
                    if (!Object.prototype.hasOwnProperty.call(aliases, alias_name)) {
                        continue;
                    }

                    /** @type{number[]} */
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
            this.searchState.descShards.set(crate, descShardList);
        }
        // Drop the (rather large) hash table used for reusing function items
        this.TYPES_POOL = new Map();
        return searchIndex;
    }

    /**
     * Parses the query.
     *
     * The supported syntax by this parser is given in the rustdoc book chapter
     * /src/doc/rustdoc/src/read-documentation/search.md
     *
     * When adding new things to the parser, add them there, too!
     *
     * @param  {string} userQuery - The user query
     *
     * @return {rustdoc.ParsedQuery<rustdoc.ParserQueryElement>} - The parsed query
     */
    static parseQuery(userQuery) {
        /**
         * @param {string} typename
         * @returns {number}
         */
        function itemTypeFromName(typename) {
            const index = itemTypes.findIndex(i => i === typename);
            if (index < 0) {
                throw ["Unknown type filter ", typename];
            }
            return index;
        }

        /**
         * @param {rustdoc.ParserQueryElement} elem
         */
        function convertTypeFilterOnElem(elem) {
            if (typeof elem.typeFilter === "string") {
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

        /**
         * Takes the user search input and returns an empty `ParsedQuery`.
         *
         * @param {string} userQuery
         *
         * @return {rustdoc.ParsedQuery<rustdoc.ParserQueryElement>}
         */
        function newParsedQuery(userQuery) {
            return {
                userQuery,
                elems: [],
                returned: [],
                // Total number of "top" elements (does not include generics).
                foundElems: 0,
                // Total number of elements (includes generics).
                totalElems: 0,
                literalSearch: false,
                hasReturnArrow: false,
                error: null,
                correction: null,
                proposeCorrectionFrom: null,
                proposeCorrectionTo: null,
                // bloom filter build from type ids
                typeFingerprint: new Uint32Array(4),
            };
        }

        /**
        * Parses the provided `query` input to fill `parserState`. If it encounters an error while
        * parsing `query`, it'll throw an error.
        *
        * @param {rustdoc.ParsedQuery<rustdoc.ParserQueryElement>} query
        * @param {rustdoc.ParserState} parserState
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
                            query.hasReturnArrow = true;
                            break;
                        }
                        throw ["Unexpected ", c, " (did you mean ", "->", "?)"];
                    } else if (parserState.pos > 0) {
                        throw ["Unexpected ", c, " after ",
                            parserState.userQuery[parserState.pos - 1]];
                    }
                    throw ["Unexpected ", c];
                } else if (c === " ") {
                    skipWhitespace(parserState);
                    continue;
                }
                if (!foundStopChar) {
                    /** @type String[] */
                    let extra = [];
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
                    query.hasReturnArrow = true;
                    break;
                } else {
                    parserState.pos += 1;
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
            userQuery,
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
            if (Array.isArray(err) && err.every(elem => typeof elem === "string")) {
                query.error = err;
            } else {
                // rethrow the error if it isn't a string array
                throw err;
            }

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
     * Executes the parsed query and builds a {ResultsTable}.
     *
     * @param  {rustdoc.ParsedQuery<rustdoc.ParserQueryElement>} origParsedQuery
     *     - The parsed user query
     * @param  {Object} filterCrates - Crate to search in if defined
     * @param  {string} currentCrate - Current crate, to rank results from this crate higher
     *
     * @return {Promise<rustdoc.ResultsTable>}
     */
    async execQuery(origParsedQuery, filterCrates, currentCrate) {
        /** @type {rustdoc.Results} */
        const results_others = new Map(),
            /** @type {rustdoc.Results} */
            results_in_args = new Map(),
            /** @type {rustdoc.Results} */
            results_returned = new Map();

        /** @type {rustdoc.ParsedQuery<rustdoc.QueryElement>} */
        // @ts-expect-error
        const parsedQuery = origParsedQuery;

        const queryLen =
            parsedQuery.elems.reduce((acc, next) => acc + next.pathLast.length, 0) +
            parsedQuery.returned.reduce((acc, next) => acc + next.pathLast.length, 0);
        const maxEditDistance = Math.floor(queryLen / 3);

        /**
         * @type {Map<string, number>}
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
         * @param {rustdoc.QueryElement} elem
         * @param {boolean=} isAssocType
         */
        const convertNameToId = (elem, isAssocType) => {
            const loweredName = elem.pathLast.toLowerCase();
            if (this.typeNameIdMap.has(loweredName) &&
                // @ts-expect-error
                (isAssocType || !this.typeNameIdMap.get(loweredName).assocOnly)) {
                // @ts-expect-error
                elem.id = this.typeNameIdMap.get(loweredName).id;
            } else if (!parsedQuery.literalSearch) {
                let match = null;
                let matchDist = maxEditDistance + 1;
                let matchName = "";
                for (const [name, { id, assocOnly }] of this.typeNameIdMap) {
                    const dist = Math.min(
                        editDistance(name, loweredName, maxEditDistance),
                        editDistance(name, elem.normalizedPathLast, maxEditDistance),
                    );
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
                const id = genericSymbols.get(elem.normalizedPathLast);
                if (id !== undefined) {
                    elem.id = id;
                } else {
                    elem.id = -(genericSymbols.size + 1);
                    genericSymbols.set(elem.normalizedPathLast, elem.id);
                }
                if (elem.typeFilter === -1 && elem.normalizedPathLast.length >= 3) {
                    // Silly heuristic to catch if the user probably meant
                    // to not write a generic parameter. We don't use it,
                    // just bring it up.
                    const maxPartDistance = Math.floor(elem.normalizedPathLast.length / 3);
                    let matchDist = maxPartDistance + 1;
                    let matchName = "";
                    for (const name of this.typeNameIdMap.keys()) {
                        const dist = editDistance(
                            name,
                            elem.normalizedPathLast,
                            maxPartDistance,
                        );
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
                    // @ts-expect-error
                    if (!this.typeNameIdMap.has(name)) {
                        parsedQuery.error = [
                            "Type parameter ",
                            // @ts-expect-error
                            name,
                            " does not exist",
                        ];
                        return [0, []];
                    }
                    for (const elem2 of constraints) {
                        convertNameToId(elem2, false);
                    }

                    // @ts-expect-error
                    return [this.typeNameIdMap.get(name).id, constraints];
                }),
            );
        };

        for (const elem of parsedQuery.elems) {
            convertNameToId(elem, false);
            this.buildFunctionTypeFingerprint(elem, parsedQuery.typeFingerprint);
        }
        for (const elem of parsedQuery.returned) {
            convertNameToId(elem, false);
            this.buildFunctionTypeFingerprint(elem, parsedQuery.typeFingerprint);
        }


        /**
         * Creates the query results.
         *
         * @param {Array<rustdoc.ResultObject>} results_in_args
         * @param {Array<rustdoc.ResultObject>} results_returned
         * @param {Array<rustdoc.ResultObject>} results_others
         * @param {rustdoc.ParsedQuery<rustdoc.QueryElement>} parsedQuery
         *
         * @return {rustdoc.ResultsTable}
         */
        function createQueryResults(
            results_in_args,
            results_returned,
            results_others,
            parsedQuery) {
            return {
                "in_args": results_in_args,
                "returned": results_returned,
                "others": results_others,
                "query": parsedQuery,
            };
        }

        // @ts-expect-error
        const buildHrefAndPath = item => {
            let displayPath;
            let href;
            const type = itemTypes[item.ty];
            const name = item.name;
            let path = item.path;
            let exactPath = item.exactPath;

            if (type === "mod") {
                displayPath = path + "::";
                href = this.rootPath + path.replace(/::/g, "/") + "/" +
                    name + "/index.html";
            } else if (type === "import") {
                displayPath = item.path + "::";
                href = this.rootPath + item.path.replace(/::/g, "/") +
                    "/index.html#reexport." + name;
            } else if (type === "primitive" || type === "keyword") {
                displayPath = "";
                exactPath = "";
                href = this.rootPath + path.replace(/::/g, "/") +
                    "/" + type + "." + name + ".html";
            } else if (type === "externcrate") {
                displayPath = "";
                href = this.rootPath + name + "/index.html";
            } else if (item.parent !== undefined) {
                const myparent = item.parent;
                let anchor = type + "." + name;
                const parentType = itemTypes[myparent.ty];
                let pageType = parentType;
                let pageName = myparent.name;
                exactPath = `${myparent.exactPath}::${myparent.name}`;

                if (parentType === "primitive") {
                    displayPath = myparent.name + "::";
                    exactPath = myparent.name;
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
                href = this.rootPath + path.replace(/::/g, "/") +
                    "/" + pageType +
                    "." + pageName +
                    ".html#" + anchor;
            } else {
                displayPath = item.path + "::";
                href = this.rootPath + item.path.replace(/::/g, "/") +
                    "/" + type + "." + name + ".html";
            }
            return [displayPath, href, `${exactPath}::${name}`];
        };

        /**
         *
         * @param {string} path
         * @returns {string}
         */
        function pathSplitter(path) {
            const tmp = "<span>" + path.replace(/::/g, "::</span><span>");
            if (tmp.endsWith("<span>")) {
                return tmp.slice(0, tmp.length - 6);
            }
            return tmp;
        }

        /**
         * Add extra data to result objects, and filter items that have been
         * marked for removal.
         *
         * @param {rustdoc.ResultObject[]} results
         * @param {"sig"|"elems"|"returned"|null} typeInfo
         * @returns {rustdoc.ResultObject[]}
         */
        const transformResults = (results, typeInfo) => {
            const duplicates = new Set();
            const out = [];

            for (const result of results) {
                if (result.id !== -1) {
                    const res = buildHrefAndPath(this.searchIndex[result.id]);
                    // many of these properties don't strictly need to be
                    // copied over, but copying them over satisfies tsc,
                    // and hopefully plays nice with the shape optimization
                    // of the browser engine.
                    /** @type {rustdoc.ResultObject} */
                    const obj = Object.assign({
                        parent: result.parent,
                        type: result.type,
                        dist: result.dist,
                        path_dist: result.path_dist,
                        index: result.index,
                        desc: result.desc,
                        item: result.item,
                        displayPath: pathSplitter(res[0]),
                        fullPath: "",
                        href: "",
                        displayTypeSignature: null,
                    }, this.searchIndex[result.id]);

                    // To be sure than it some items aren't considered as duplicate.
                    obj.fullPath = res[2] + "|" + obj.ty;

                    if (duplicates.has(obj.fullPath)) {
                        continue;
                    }

                    // Exports are specifically not shown if the items they point at
                    // are already in the results.
                    if (obj.ty === TY_IMPORT && duplicates.has(res[2])) {
                        continue;
                    }
                    if (duplicates.has(res[2] + "|" + TY_IMPORT)) {
                        continue;
                    }
                    duplicates.add(obj.fullPath);
                    duplicates.add(res[2]);

                    if (typeInfo !== null) {
                        obj.displayTypeSignature =
                            // @ts-expect-error
                            this.formatDisplayTypeSignature(obj, typeInfo);
                    }

                    obj.href = res[1];
                    out.push(obj);
                    if (out.length >= MAX_RESULTS) {
                        break;
                    }
                }
            }
            return out;
        };

        /**
         * Add extra data to result objects, and filter items that have been
         * marked for removal.
         *
         * The output is formatted as an array of hunks, where odd numbered
         * hunks are highlighted and even numbered ones are not.
         *
         * @param {rustdoc.ResultObject} obj
         * @param {"sig"|"elems"|"returned"|null} typeInfo
         * @returns {Promise<rustdoc.DisplayTypeSignature>}
         */
        this.formatDisplayTypeSignature = async(obj, typeInfo) => {
            const objType = obj.type;
            if (!objType) {
                return {type: [], mappedNames: new Map(), whereClause: new Map()};
            }
            let fnInputs = null;
            let fnOutput = null;
            /** @type {Map<number, number> | null} */
            let mgens = null;
            if (typeInfo !== "elems" && typeInfo !== "returned") {
                fnInputs = unifyFunctionTypes(
                    objType.inputs,
                    parsedQuery.elems,
                    objType.where_clause,
                    null,
                    mgensScratch => {
                        fnOutput = unifyFunctionTypes(
                            objType.output,
                            parsedQuery.returned,
                            objType.where_clause,
                            mgensScratch,
                            mgensOut => {
                                mgens = mgensOut;
                                return true;
                            },
                            0,
                        );
                        return !!fnOutput;
                    },
                    0,
                );
            } else {
                const arr = typeInfo === "elems" ? objType.inputs : objType.output;
                const highlighted = unifyFunctionTypes(
                    arr,
                    parsedQuery.elems,
                    objType.where_clause,
                    null,
                    mgensOut => {
                        mgens = mgensOut;
                        return true;
                    },
                    0,
                );
                if (typeInfo === "elems") {
                    fnInputs = highlighted;
                } else {
                    fnOutput = highlighted;
                }
            }
            if (!fnInputs) {
                fnInputs = objType.inputs;
            }
            if (!fnOutput) {
                fnOutput = objType.output;
            }
            const mappedNames = new Map();
            const whereClause = new Map();

            const fnParamNames = obj.paramNames || [];
            /** @type {string[]} */
            const queryParamNames = [];
            /**
             * Recursively writes a map of IDs to query generic names,
             * which are later used to map query generic names to function generic names.
             * For example, when the user writes `X -> Option<X>` and the function
             * is actually written as `T -> Option<T>`, this function stores the
             * mapping `(-1, "X")`, and the writeFn function looks up the entry
             * for -1 to form the final, user-visible mapping of "X is T".
             *
             * @param {rustdoc.QueryElement} queryElem
             */
            const remapQuery = queryElem => {
                if (queryElem.id !== null && queryElem.id < 0) {
                    queryParamNames[-1 - queryElem.id] = queryElem.name;
                }
                if (queryElem.generics.length > 0) {
                    queryElem.generics.forEach(remapQuery);
                }
                if (queryElem.bindings.size > 0) {
                    [...queryElem.bindings.values()].flat().forEach(remapQuery);
                }
            };

            parsedQuery.elems.forEach(remapQuery);
            parsedQuery.returned.forEach(remapQuery);

            /**
             * Write text to a highlighting array.
             * Index 0 is not highlighted, index 1 is highlighted,
             * index 2 is not highlighted, etc.
             *
             * @param {{name?: string, highlighted?: boolean}} fnType - input
             * @param {string[]} result
             */
            const pushText = (fnType, result) => {
                // If !!(result.length % 2) == false, then pushing a new slot starts an even
                // numbered slot. Even numbered slots are not highlighted.
                //
                // `highlighted` will not be defined if an entire subtree is not highlighted,
                // so `!!` is used to coerce it to boolean. `result.length % 2` is used to
                // check if the number is even, but it evaluates to a number, so it also
                // needs coerced to a boolean.
                if (!!(result.length % 2) === !!fnType.highlighted) {
                    result.push("");
                } else if (result.length === 0 && !!fnType.highlighted) {
                    result.push("");
                    result.push("");
                }

                result[result.length - 1] += fnType.name;
            };

            /**
             * Write a higher order function type: either a function pointer
             * or a trait bound on Fn, FnMut, or FnOnce.
             *
             * @param {rustdoc.HighlightedFunctionType} fnType - input
             * @param {string[]} result
             */
            const writeHof = (fnType, result) => {
                const hofOutput = fnType.bindings.get(this.typeNameIdOfOutput) || [];
                const hofInputs = fnType.generics;
                pushText(fnType, result);
                pushText({name: " (", highlighted: false}, result);
                let needsComma = false;
                for (const fnType of hofInputs) {
                    if (needsComma) {
                        pushText({ name: ", ", highlighted: false }, result);
                    }
                    needsComma = true;
                    writeFn(fnType, result);
                }
                pushText({
                    name: hofOutput.length === 0 ? ")" : ") -> ",
                    highlighted: false,
                }, result);
                if (hofOutput.length > 1) {
                    pushText({name: "(", highlighted: false}, result);
                }
                needsComma = false;
                for (const fnType of hofOutput) {
                    if (needsComma) {
                        pushText({ name: ", ", highlighted: false }, result);
                    }
                    needsComma = true;
                    writeFn(fnType, result);
                }
                if (hofOutput.length > 1) {
                    pushText({name: ")", highlighted: false}, result);
                }
            };

            /**
             * Write a primitive type with special syntax, like `!` or `[T]`.
             * Returns `false` if the supplied type isn't special.
             *
             * @param {rustdoc.HighlightedFunctionType} fnType
             * @param {string[]} result
             */
            const writeSpecialPrimitive = (fnType, result) => {
                if (fnType.id === this.typeNameIdOfArray || fnType.id === this.typeNameIdOfSlice ||
                    fnType.id === this.typeNameIdOfTuple || fnType.id === this.typeNameIdOfUnit) {
                    const [ob, sb] =
                        fnType.id === this.typeNameIdOfArray ||
                            fnType.id === this.typeNameIdOfSlice ?
                        ["[", "]"] :
                        ["(", ")"];
                    pushText({ name: ob, highlighted: fnType.highlighted }, result);
                    onEachBtwn(
                        fnType.generics,
                        nested => writeFn(nested, result),
                        // @ts-expect-error
                        () => pushText({ name: ", ", highlighted: false }, result),
                    );
                    pushText({ name: sb, highlighted: fnType.highlighted }, result);
                    return true;
                } else if (fnType.id === this.typeNameIdOfReference) {
                    pushText({ name: "&", highlighted: fnType.highlighted }, result);
                    let prevHighlighted = false;
                    onEachBtwn(
                        fnType.generics,
                        value => {
                            prevHighlighted = !!value.highlighted;
                            writeFn(value, result);
                        },
                        // @ts-expect-error
                        value => pushText({
                            name: " ",
                            highlighted: prevHighlighted && value.highlighted,
                        }, result),
                    );
                    return true;
                } else if (fnType.id === this.typeNameIdOfFn) {
                    writeHof(fnType, result);
                    return true;
                }
                return false;
            };
            /**
             * Write a type. This function checks for special types,
             * like slices, with their own formatting. It also handles
             * updating the where clause and generic type param map.
             *
             * @param {rustdoc.HighlightedFunctionType} fnType
             * @param {string[]} result
             */
            const writeFn = (fnType, result) => {
                if (fnType.id !== null && fnType.id < 0) {
                    if (fnParamNames[-1 - fnType.id] === "") {
                        // Normally, there's no need to shown an unhighlighted
                        // where clause, but if it's impl Trait, then we do.
                        const generics = fnType.generics.length > 0 ?
                            fnType.generics :
                            objType.where_clause[-1 - fnType.id];
                        for (const nested of generics) {
                            writeFn(nested, result);
                        }
                        return;
                    } else if (mgens) {
                        for (const [queryId, fnId] of mgens) {
                            if (fnId === fnType.id) {
                                mappedNames.set(
                                    queryParamNames[-1 - queryId],
                                    fnParamNames[-1 - fnType.id],
                                );
                            }
                        }
                    }
                    pushText({
                        name: fnParamNames[-1 - fnType.id],
                        highlighted: !!fnType.highlighted,
                    }, result);
                    /** @type{string[]} */
                    const where = [];
                    onEachBtwn(
                        fnType.generics,
                        nested => writeFn(nested, where),
                        // @ts-expect-error
                        () => pushText({ name: " + ", highlighted: false }, where),
                    );
                    if (where.length > 0) {
                        whereClause.set(fnParamNames[-1 - fnType.id], where);
                    }
                } else {
                    if (fnType.ty === TY_PRIMITIVE) {
                        if (writeSpecialPrimitive(fnType, result)) {
                            return;
                        }
                    } else if (fnType.ty === TY_TRAIT && (
                        fnType.id === this.typeNameIdOfFn ||
                            fnType.id === this.typeNameIdOfFnMut ||
                            fnType.id === this.typeNameIdOfFnOnce)) {
                        writeHof(fnType, result);
                        return;
                    }
                    pushText(fnType, result);
                    let hasBindings = false;
                    if (fnType.bindings.size > 0) {
                        onEachBtwn(
                            fnType.bindings,
                            ([key, values]) => {
                                const name = this.assocTypeIdNameMap.get(key);
                                // @ts-expect-error
                                if (values.length === 1 && values[0].id < 0 &&
                                    // @ts-expect-error
                                    `${fnType.name}::${name}` === fnParamNames[-1 - values[0].id]) {
                                    // the internal `Item=Iterator::Item` type variable should be
                                    // shown in the where clause and name mapping output, but is
                                    // redundant in this spot
                                    for (const value of values) {
                                        writeFn(value, []);
                                    }
                                    return true;
                                }
                                if (!hasBindings) {
                                    hasBindings = true;
                                    pushText({ name: "<", highlighted: false }, result);
                                }
                                pushText({ name, highlighted: false }, result);
                                pushText({
                                    name: values.length !== 1 ? "=(" : "=",
                                    highlighted: false,
                                }, result);
                                onEachBtwn(
                                    values || [],
                                    value => writeFn(value, result),
                                    // @ts-expect-error
                                    () => pushText({ name: " + ",  highlighted: false }, result),
                                );
                                if (values.length !== 1) {
                                    pushText({ name: ")", highlighted: false }, result);
                                }
                            },
                            // @ts-expect-error
                            () => pushText({ name: ", ",  highlighted: false }, result),
                        );
                    }
                    if (fnType.generics.length > 0) {
                        pushText({ name: hasBindings ? ", " : "<", highlighted: false }, result);
                    }
                    onEachBtwn(
                        fnType.generics,
                        value => writeFn(value, result),
                        // @ts-expect-error
                        () => pushText({ name: ", ",  highlighted: false }, result),
                    );
                    if (hasBindings || fnType.generics.length > 0) {
                        pushText({ name: ">", highlighted: false }, result);
                    }
                }
            };
            /** @type {string[]} */
            const type = [];
            onEachBtwn(
                fnInputs,
                fnType => writeFn(fnType, type),
                // @ts-expect-error
                () => pushText({ name: ", ",  highlighted: false }, type),
            );
            pushText({ name: " -> ", highlighted: false }, type);
            onEachBtwn(
                fnOutput,
                fnType => writeFn(fnType, type),
                // @ts-expect-error
                () => pushText({ name: ", ",  highlighted: false }, type),
            );

            return {type, mappedNames, whereClause};
        };

        /**
         * This function takes a result map, and sorts it by various criteria, including edit
         * distance, substring match, and the crate it comes from.
         *
         * @param {rustdoc.Results} results
         * @param {"sig"|"elems"|"returned"|null} typeInfo
         * @param {string} preferredCrate
         * @returns {Promise<rustdoc.ResultObject[]>}
         */
        const sortResults = async(results, typeInfo, preferredCrate) => {
            const userQuery = parsedQuery.userQuery;
            const normalizedUserQuery = parsedQuery.userQuery.toLowerCase();
            const isMixedCase = normalizedUserQuery !== userQuery;
            const result_list = [];
            const isReturnTypeQuery = parsedQuery.elems.length === 0 ||
                typeInfo === "returned";
            for (const result of results.values()) {
                result.item = this.searchIndex[result.id];
                result.word = this.searchIndex[result.id].word;
                if (isReturnTypeQuery) {
                    // we are doing a return-type based search,
                    // deprioritize "clone-like" results,
                    // ie. functions that also take the queried type as an argument.
                    const resultItemType = result.item && result.item.type;
                    if (!resultItemType) {
                        continue;
                    }
                    const inputs = resultItemType.inputs;
                    const where_clause = resultItemType.where_clause;
                    if (containsTypeFromQuery(inputs, where_clause)) {
                        result.path_dist *= 100;
                        result.dist *= 100;
                    }
                }
                result_list.push(result);
            }

            result_list.sort((aaa, bbb) => {
                /** @type {number} */
                let a;
                /** @type {number} */
                let b;

                // sort by exact case-sensitive match
                if (isMixedCase) {
                    a = Number(aaa.item.name !== userQuery);
                    b = Number(bbb.item.name !== userQuery);
                    if (a !== b) {
                        return a - b;
                    }
                }

                // sort by exact match with regard to the last word (mismatch goes later)
                a = Number(aaa.word !== normalizedUserQuery);
                b = Number(bbb.word !== normalizedUserQuery);
                if (a !== b) {
                    return a - b;
                }

                // sort by index of keyword in item name (no literal occurrence goes later)
                a = Number(aaa.index < 0);
                b = Number(bbb.index < 0);
                if (a !== b) {
                    return a - b;
                }

                // in type based search, put functions first
                if (parsedQuery.hasReturnArrow) {
                    a = Number(!isFnLikeTy(aaa.item.ty));
                    b = Number(!isFnLikeTy(bbb.item.ty));
                    if (a !== b) {
                        return a - b;
                    }
                }

                // Sort by distance in the path part, if specified
                // (less changes required to match means higher rankings)
                a = Number(aaa.path_dist);
                b = Number(bbb.path_dist);
                if (a !== b) {
                    return a - b;
                }

                // (later literal occurrence, if any, goes later)
                a = Number(aaa.index);
                b = Number(bbb.index);
                if (a !== b) {
                    return a - b;
                }

                // Sort by distance in the name part, the last part of the path
                // (less changes required to match means higher rankings)
                a = Number(aaa.dist);
                b = Number(bbb.dist);
                if (a !== b) {
                    return a - b;
                }

                // sort deprecated items later
                a = Number(
                    // @ts-expect-error
                    this.searchIndexDeprecated.get(aaa.item.crate).contains(aaa.item.bitIndex),
                );
                b = Number(
                    // @ts-expect-error
                    this.searchIndexDeprecated.get(bbb.item.crate).contains(bbb.item.bitIndex),
                );
                if (a !== b) {
                    return a - b;
                }

                // sort by crate (current crate comes first)
                a = Number(aaa.item.crate !== preferredCrate);
                b = Number(bbb.item.crate !== preferredCrate);
                if (a !== b) {
                    return a - b;
                }

                // sort by item name length (longer goes later)
                a = Number(aaa.word.length);
                b = Number(bbb.word.length);
                if (a !== b) {
                    return a - b;
                }

                // sort by item name (lexicographically larger goes later)
                let aw = aaa.word;
                let bw = bbb.word;
                if (aw !== bw) {
                    return (aw > bw ? +1 : -1);
                }

                // sort by description (no description goes later)
                a = Number(
                    // @ts-expect-error
                    this.searchIndexEmptyDesc.get(aaa.item.crate).contains(aaa.item.bitIndex),
                );
                b = Number(
                    // @ts-expect-error
                    this.searchIndexEmptyDesc.get(bbb.item.crate).contains(bbb.item.bitIndex),
                );
                if (a !== b) {
                    return a - b;
                }

                // sort by type (later occurrence in `itemTypes` goes later)
                a = Number(aaa.item.ty);
                b = Number(bbb.item.ty);
                if (a !== b) {
                    return a - b;
                }

                // sort by path (lexicographically larger goes later)
                aw = aaa.item.path;
                bw = bbb.item.path;
                if (aw !== bw) {
                    return (aw > bw ? +1 : -1);
                }

                // que sera, sera
                return 0;
            });

            return transformResults(result_list, typeInfo);
        };

        /**
         * This function checks if a list of search query `queryElems` can all be found in the
         * search index (`fnTypes`).
         *
         * This function returns highlighted results on a match, or `null`. If `solutionCb` is
         * supplied, it will call that function with mgens, and that callback can accept or
         * reject the result by returning `true` or `false`. If the callback returns false,
         * then this function will try with a different solution, or bail with null if it
         * runs out of candidates.
         *
         * @param {rustdoc.FunctionType[]} fnTypesIn - The objects to check.
         * @param {rustdoc.QueryElement[]} queryElems - The elements from the parsed query.
         * @param {rustdoc.FunctionType[][]} whereClause - Trait bounds for generic items.
         * @param {Map<number,number>|null} mgensIn
         *     - Map query generics to function generics (never modified).
         * @param {function(Map<number,number>?): boolean} solutionCb
         *     - Called for each `mgens` solution.
         * @param {number} unboxingDepth
         *     - Limit checks that Ty matches Vec<Ty>,
         *       but not Vec<ParamEnvAnd<WithInfcx<ConstTy<Interner<Ty=Ty>>>>>
         *
         * @return {rustdoc.HighlightedFunctionType[]|null}
         *     - Returns highlighted results if a match, null otherwise.
         */
        function unifyFunctionTypes(
            fnTypesIn,
            queryElems,
            whereClause,
            mgensIn,
            solutionCb,
            unboxingDepth,
        ) {
            if (unboxingDepth >= UNBOXING_LIMIT) {
                return null;
            }
            /**
             * @type {Map<number, number>|null}
             */
            const mgens = mgensIn === null ? null : new Map(mgensIn);
            if (queryElems.length === 0) {
                return solutionCb(mgens) ? fnTypesIn : null;
            }
            if (!fnTypesIn || fnTypesIn.length === 0) {
                return null;
            }
            const ql = queryElems.length;
            const fl = fnTypesIn.length;

            // One element fast path / base case
            if (ql === 1 && queryElems[0].generics.length === 0
                && queryElems[0].bindings.size === 0) {
                const queryElem = queryElems[0];
                for (const [i, fnType] of fnTypesIn.entries()) {
                    if (!unifyFunctionTypeIsMatchCandidate(fnType, queryElem, mgens)) {
                        continue;
                    }
                    if (fnType.id !== null &&
                        fnType.id < 0 &&
                        queryElem.id !== null &&
                        queryElem.id < 0
                    ) {
                        if (mgens && mgens.has(queryElem.id) &&
                            mgens.get(queryElem.id) !== fnType.id) {
                            continue;
                        }
                        const mgensScratch = new Map(mgens);
                        mgensScratch.set(queryElem.id, fnType.id);
                        if (!solutionCb || solutionCb(mgensScratch)) {
                            const highlighted = [...fnTypesIn];
                            highlighted[i] = Object.assign({
                                highlighted: true,
                            }, fnType, {
                                generics: whereClause[-1 - fnType.id],
                            });
                            return highlighted;
                        }
                    } else if (solutionCb(mgens ? new Map(mgens) : null)) {
                        // unifyFunctionTypeIsMatchCandidate already checks that ids match
                        const highlighted = [...fnTypesIn];
                        highlighted[i] = Object.assign({
                            highlighted: true,
                        }, fnType, {
                            generics: unifyGenericTypes(
                                fnType.generics,
                                queryElem.generics,
                                whereClause,
                                mgens ? new Map(mgens) : null,
                                solutionCb,
                                unboxingDepth,
                            ) || fnType.generics,
                        });
                        return highlighted;
                    }
                }
                for (const [i, fnType] of fnTypesIn.entries()) {
                    if (!unifyFunctionTypeIsUnboxCandidate(
                        fnType,
                        queryElem,
                        whereClause,
                        mgens,
                        unboxingDepth + 1,
                    )) {
                        continue;
                    }
                    // @ts-expect-error
                    if (fnType.id < 0) {
                        const highlightedGenerics = unifyFunctionTypes(
                            // @ts-expect-error
                            whereClause[(-fnType.id) - 1],
                            queryElems,
                            whereClause,
                            mgens,
                            solutionCb,
                            unboxingDepth + 1,
                        );
                        if (highlightedGenerics) {
                            const highlighted = [...fnTypesIn];
                            highlighted[i] = Object.assign({
                                highlighted: true,
                            }, fnType, {
                                generics: highlightedGenerics,
                            });
                            return highlighted;
                        }
                    } else {
                        const highlightedGenerics = unifyFunctionTypes(
                            [...Array.from(fnType.bindings.values()).flat(), ...fnType.generics],
                            queryElems,
                            whereClause,
                            mgens ? new Map(mgens) : null,
                            solutionCb,
                            unboxingDepth + 1,
                        );
                        if (highlightedGenerics) {
                            const highlighted = [...fnTypesIn];
                            highlighted[i] = Object.assign({}, fnType, {
                                generics: highlightedGenerics,
                                bindings: new Map([...fnType.bindings.entries()].map(([k, v]) => {
                                    return [k, highlightedGenerics.splice(0, v.length)];
                                })),
                            });
                            return highlighted;
                        }
                    }
                }
                return null;
            }

            // Multiple element recursive case
            /**
             * @type {Array<rustdoc.FunctionType>}
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
                if (fnType.id !== null && queryElem.id !== null && fnType.id < 0) {
                    mgensScratch = new Map(mgens);
                    if (mgensScratch.has(queryElem.id)
                        && mgensScratch.get(queryElem.id) !== fnType.id) {
                        continue;
                    }
                    mgensScratch.set(queryElem.id, fnType.id);
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
                /** @type {rustdoc.HighlightedFunctionType[]|null} */
                let unifiedGenerics = [];
                /** @type {null|Map<number, number>} */
                let unifiedGenericsMgens = null;
                /** @type {rustdoc.HighlightedFunctionType[]|null} */
                const passesUnification = unifyFunctionTypes(
                    fnTypes,
                    queryElemsTmp,
                    whereClause,
                    mgensScratch,
                    mgensScratch => {
                        if (fnType.generics.length === 0 && queryElem.generics.length === 0
                            && fnType.bindings.size === 0 && queryElem.bindings.size === 0) {
                            return solutionCb(mgensScratch);
                        }
                        const solution = unifyFunctionTypeCheckBindings(
                            fnType,
                            queryElem,
                            whereClause,
                            mgensScratch,
                            unboxingDepth,
                        );
                        if (!solution) {
                            return false;
                        }
                        const simplifiedGenerics = solution.simplifiedGenerics;
                        for (const simplifiedMgens of solution.mgens) {
                            unifiedGenerics = unifyGenericTypes(
                                simplifiedGenerics,
                                queryElem.generics,
                                whereClause,
                                simplifiedMgens,
                                solutionCb,
                                unboxingDepth,
                            );
                            if (unifiedGenerics !== null) {
                                unifiedGenericsMgens = simplifiedMgens;
                                return true;
                            }
                        }
                        return false;
                    },
                    unboxingDepth,
                );
                if (passesUnification) {
                    passesUnification.length = fl;
                    passesUnification[flast] = passesUnification[i];
                    passesUnification[i] = Object.assign({}, fnType, {
                        highlighted: true,
                        generics: unifiedGenerics,
                        bindings: new Map([...fnType.bindings.entries()].map(([k, v]) => {
                            return [k, queryElem.bindings.has(k) ? unifyFunctionTypes(
                                v,
                                // @ts-expect-error
                                queryElem.bindings.get(k),
                                whereClause,
                                unifiedGenericsMgens,
                                solutionCb,
                                unboxingDepth,
                            // @ts-expect-error
                            ) : unifiedGenerics.splice(0, v.length)];
                        })),
                    });
                    return passesUnification;
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
                    unboxingDepth + 1,
                )) {
                    continue;
                }
                const generics = fnType.id !== null && fnType.id < 0 ?
                    whereClause[(-fnType.id) - 1] :
                    fnType.generics;
                const bindings = fnType.bindings ?
                    Array.from(fnType.bindings.values()).flat() :
                    [];
                const passesUnification = unifyFunctionTypes(
                    fnTypes.toSpliced(i, 1, ...bindings, ...generics),
                    queryElems,
                    whereClause,
                    mgens,
                    solutionCb,
                    unboxingDepth + 1,
                );
                if (passesUnification) {
                    const highlightedGenerics = passesUnification.slice(
                        i,
                        i + generics.length + bindings.length,
                    );
                    const highlightedFnType = Object.assign({}, fnType, {
                        generics: highlightedGenerics,
                        bindings: new Map([...fnType.bindings.entries()].map(([k, v]) => {
                            return [k, highlightedGenerics.splice(0, v.length)];
                        })),
                    });
                    return passesUnification.toSpliced(
                        i,
                        generics.length + bindings.length,
                        highlightedFnType,
                    );
                }
            }
            return null;
        }
        /**
         * This function compares two lists of generics.
         *
         * This function behaves very similarly to `unifyFunctionTypes`, except that it
         * doesn't skip or reorder anything. This is intended to match the behavior of
         * the ordinary Rust type system, so that `Vec<Allocator>` only matches an actual
         * `Vec` of `Allocators` and not the implicit `Allocator` parameter that every
         * `Vec` has.
         *
         * @param {Array<rustdoc.FunctionType>} fnTypesIn - The objects to check.
         * @param {Array<rustdoc.QueryElement>} queryElems - The elements from the parsed query.
         * @param {rustdoc.FunctionType[][]} whereClause - Trait bounds for generic items.
         * @param {Map<number,number>|null} mgensIn
         *     - Map functions generics to query generics (never modified).
         * @param {function(Map<number,number>): boolean} solutionCb
         *     - Called for each `mgens` solution.
         * @param {number} unboxingDepth
         *     - Limit checks that Ty matches Vec<Ty>,
         *       but not Vec<ParamEnvAnd<WithInfcx<ConstTy<Interner<Ty=Ty>>>>>
         *
         * @return {rustdoc.HighlightedFunctionType[]|null}
         *     - Returns highlighted results if a match, null otherwise.
         */
        function unifyGenericTypes(
            fnTypesIn,
            queryElems,
            whereClause,
            mgensIn,
            solutionCb,
            unboxingDepth,
        ) {
            if (unboxingDepth >= UNBOXING_LIMIT) {
                return null;
            }
            /**
             * @type {Map<number, number>|null}
             */
            const mgens = mgensIn === null ? null : new Map(mgensIn);
            if (queryElems.length === 0) {
                // @ts-expect-error
                return solutionCb(mgens) ? fnTypesIn : null;
            }
            if (!fnTypesIn || fnTypesIn.length === 0) {
                return null;
            }
            const fnType = fnTypesIn[0];
            const queryElem = queryElems[0];
            if (unifyFunctionTypeIsMatchCandidate(fnType, queryElem, mgens)) {
                if (fnType.id !== null &&
                    fnType.id < 0 &&
                    queryElem.id !== null &&
                    queryElem.id < 0
                ) {
                    if (!mgens || !mgens.has(queryElem.id) ||
                        mgens.get(queryElem.id) === fnType.id
                    ) {
                        const mgensScratch = new Map(mgens);
                        mgensScratch.set(queryElem.id, fnType.id);
                        const fnTypesRemaining = unifyGenericTypes(
                            fnTypesIn.slice(1),
                            queryElems.slice(1),
                            whereClause,
                            mgensScratch,
                            solutionCb,
                            unboxingDepth,
                        );
                        if (fnTypesRemaining) {
                            const highlighted = [fnType, ...fnTypesRemaining];
                            highlighted[0] = Object.assign({
                                highlighted: true,
                            }, fnType, {
                                generics: whereClause[-1 - fnType.id],
                            });
                            return highlighted;
                        }
                    }
                } else {
                    let unifiedGenerics;
                    const fnTypesRemaining = unifyGenericTypes(
                        fnTypesIn.slice(1),
                        queryElems.slice(1),
                        whereClause,
                        mgens,
                        // @ts-expect-error
                        mgensScratch => {
                            const solution = unifyFunctionTypeCheckBindings(
                                fnType,
                                queryElem,
                                whereClause,
                                mgensScratch,
                                unboxingDepth,
                            );
                            if (!solution) {
                                return false;
                            }
                            const simplifiedGenerics = solution.simplifiedGenerics;
                            for (const simplifiedMgens of solution.mgens) {
                                unifiedGenerics = unifyGenericTypes(
                                    simplifiedGenerics,
                                    queryElem.generics,
                                    whereClause,
                                    simplifiedMgens,
                                    solutionCb,
                                    unboxingDepth,
                                );
                                if (unifiedGenerics !== null) {
                                    return true;
                                }
                            }
                        },
                        unboxingDepth,
                    );
                    if (fnTypesRemaining) {
                        const highlighted = [fnType, ...fnTypesRemaining];
                        highlighted[0] = Object.assign({
                            highlighted: true,
                        }, fnType, {
                            generics: unifiedGenerics || fnType.generics,
                        });
                        return highlighted;
                    }
                }
            }
            if (unifyFunctionTypeIsUnboxCandidate(
                fnType,
                queryElem,
                whereClause,
                mgens,
                unboxingDepth + 1,
            )) {
                let highlightedRemaining;
                if (fnType.id !== null && fnType.id < 0) {
                    // Where clause corresponds to `F: A + B`
                    //                                 ^^^^^
                    // The order of the constraints doesn't matter, so
                    // use order-agnostic matching for it.
                    const highlightedGenerics = unifyFunctionTypes(
                        whereClause[(-fnType.id) - 1],
                        [queryElem],
                        whereClause,
                        mgens,
                        // @ts-expect-error
                        mgensScratch => {
                            const hl = unifyGenericTypes(
                                fnTypesIn.slice(1),
                                queryElems.slice(1),
                                whereClause,
                                mgensScratch,
                                solutionCb,
                                unboxingDepth,
                            );
                            if (hl) {
                                highlightedRemaining = hl;
                            }
                            return hl;
                        },
                        unboxingDepth + 1,
                    );
                    if (highlightedGenerics) {
                        return [Object.assign({
                            highlighted: true,
                        }, fnType, {
                            generics: highlightedGenerics,
                        // @ts-expect-error
                        }), ...highlightedRemaining];
                    }
                } else {
                    const highlightedGenerics = unifyGenericTypes(
                        [
                            ...Array.from(fnType.bindings.values()).flat(),
                            ...fnType.generics,
                        ],
                        [queryElem],
                        whereClause,
                        mgens,
                        // @ts-expect-error
                        mgensScratch => {
                            const hl = unifyGenericTypes(
                                fnTypesIn.slice(1),
                                queryElems.slice(1),
                                whereClause,
                                mgensScratch,
                                solutionCb,
                                unboxingDepth,
                            );
                            if (hl) {
                                highlightedRemaining = hl;
                            }
                            return hl;
                        },
                        unboxingDepth + 1,
                    );
                    if (highlightedGenerics) {
                        return [Object.assign({}, fnType, {
                            generics: highlightedGenerics,
                            bindings: new Map([...fnType.bindings.entries()].map(([k, v]) => {
                                return [k, highlightedGenerics.splice(0, v.length)];
                            })),
                        // @ts-expect-error
                        }), ...highlightedRemaining];
                    }
                }
            }
            return null;
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
         * @param {rustdoc.FunctionType} fnType
         * @param {rustdoc.QueryElement} queryElem
         * @param {Map<number,number>|null} mgensIn - Map query generics to function generics.
         * @returns {boolean}
         */
        const unifyFunctionTypeIsMatchCandidate = (fnType, queryElem, mgensIn) => {
            // type filters look like `trait:Read` or `enum:Result`
            if (!typePassesFilter(queryElem.typeFilter, fnType.ty)) {
                return false;
            }
            // fnType.id < 0 means generic
            // queryElem.id < 0 does too
            // mgensIn[queryElem.id] = fnType.id
            if (fnType.id !== null && fnType.id < 0 && queryElem.id !== null && queryElem.id < 0) {
                if (
                    mgensIn && mgensIn.has(queryElem.id) &&
                    mgensIn.get(queryElem.id) !== fnType.id
                ) {
                    return false;
                }
                return true;
            } else {
                if (queryElem.id === this.typeNameIdOfArrayOrSlice &&
                    (fnType.id === this.typeNameIdOfSlice || fnType.id === this.typeNameIdOfArray)
                ) {
                    // [] matches primitive:array or primitive:slice
                    // if it matches, then we're fine, and this is an appropriate match candidate
                } else if (queryElem.id === this.typeNameIdOfTupleOrUnit &&
                    (fnType.id === this.typeNameIdOfTuple || fnType.id === this.typeNameIdOfUnit)
                ) {
                    // () matches primitive:tuple or primitive:unit
                    // if it matches, then we're fine, and this is an appropriate match candidate
                } else if (queryElem.id === this.typeNameIdOfHof &&
                    (fnType.id === this.typeNameIdOfFn || fnType.id === this.typeNameIdOfFnMut ||
                        fnType.id === this.typeNameIdOfFnOnce)
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
        };
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
         * @param {rustdoc.FunctionType} fnType
         * @param {rustdoc.QueryElement} queryElem
         * @param {rustdoc.FunctionType[][]} whereClause - Trait bounds for generic items.
         * @param {Map<number,number>|null} mgensIn - Map query generics to function generics.
         *                                            Never modified.
         * @param {number} unboxingDepth
         * @returns {false|{
         *     mgens: [Map<number,number>|null], simplifiedGenerics: rustdoc.FunctionType[]
         * }}
         */
        function unifyFunctionTypeCheckBindings(
            fnType,
            queryElem,
            whereClause,
            mgensIn,
            unboxingDepth,
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
                        /** @type{Array<Map<number, number> | null>} */
                        const newSolutions = [];
                        unifyFunctionTypes(
                            // @ts-expect-error
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
                            unboxingDepth,
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
                    simplifiedGenerics = [...binds, ...simplifiedGenerics];
                } else {
                    simplifiedGenerics = binds;
                }
                // @ts-expect-error
                return { simplifiedGenerics, mgens: mgensSolutionSet };
            }
            return { simplifiedGenerics, mgens: [mgensIn] };
        }
        /**
         * @param {rustdoc.FunctionType} fnType
         * @param {rustdoc.QueryElement} queryElem
         * @param {rustdoc.FunctionType[][]} whereClause - Trait bounds for generic items.
         * @param {Map<number,number>|null} mgens - Map query generics to function generics.
         * @param {number} unboxingDepth
         * @returns {boolean}
         */
        function unifyFunctionTypeIsUnboxCandidate(
            fnType,
            queryElem,
            whereClause,
            mgens,
            unboxingDepth,
        ) {
            if (unboxingDepth >= UNBOXING_LIMIT) {
                return false;
            }
            if (fnType.id !== null && fnType.id < 0) {
                if (!whereClause) {
                    return false;
                }
                // This is only a potential unbox if the search query appears in the where clause
                // for example, searching `Read -> usize` should find
                // `fn read_all<R: Read>(R) -> Result<usize>`
                // generic `R` is considered "unboxed"
                return checkIfInList(
                    whereClause[(-fnType.id) - 1],
                    queryElem,
                    whereClause,
                    mgens,
                    unboxingDepth,
                );
            } else if (fnType.unboxFlag &&
                (fnType.generics.length > 0 || fnType.bindings.size > 0)) {
                const simplifiedGenerics = [
                    ...fnType.generics,
                    ...Array.from(fnType.bindings.values()).flat(),
                ];
                return checkIfInList(
                    simplifiedGenerics,
                    queryElem,
                    whereClause,
                    mgens,
                    unboxingDepth,
                );
            }
            return false;
        }

        /**
         * This function checks if the given list contains any
         * (non-generic) types mentioned in the query.
         *
         * @param {rustdoc.FunctionType[]} list    - A list of function types.
         * @param {rustdoc.FunctionType[][]} where_clause - Trait bounds for generic items.
         */
        function containsTypeFromQuery(list, where_clause) {
            if (!list) return false;
            for (const ty of parsedQuery.returned) {
                // negative type ids are generics
                if (ty.id !== null && ty.id < 0) {
                    continue;
                }
                if (checkIfInList(list, ty, where_clause, null, 0)) {
                    return true;
                }
            }
            for (const ty of parsedQuery.elems) {
                if (ty.id !== null && ty.id < 0) {
                    continue;
                }
                if (checkIfInList(list, ty, where_clause, null, 0)) {
                    return true;
                }
            }
            return false;
        }

        /**
         * This function checks if the object (`row`) matches the given type (`elem`) and its
         * generics (if any).
         *
         * @param {rustdoc.FunctionType[]} list
         * @param {rustdoc.QueryElement} elem          - The element from the parsed query.
         * @param {rustdoc.FunctionType[][]} whereClause - Trait bounds for generic items.
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
         * @param {rustdoc.FunctionType} row
         * @param {rustdoc.QueryElement} elem          - The element from the parsed query.
         * @param {rustdoc.FunctionType[][]} whereClause - Trait bounds for generic items.
         * @param {Map<number,number>|null} mgens - Map query generics to function generics.
         *
         * @return {boolean} - Returns true if the type matches, false otherwise.
         */
        // @ts-expect-error
        const checkType = (row, elem, whereClause, mgens, unboxingDepth) => {
            if (unboxingDepth >= UNBOXING_LIMIT) {
                return false;
            }
            if (row.id !== null && elem.id !== null &&
                row.id > 0 && elem.id > 0 && elem.pathWithoutLast.length === 0 &&
                row.generics.length === 0 && elem.generics.length === 0 &&
                row.bindings.size === 0 && elem.bindings.size === 0 &&
                // special case
                elem.id !== this.typeNameIdOfArrayOrSlice &&
                elem.id !== this.typeNameIdOfHof &&
                elem.id !== this.typeNameIdOfTupleOrUnit
            ) {
                return row.id === elem.id && typePassesFilter(elem.typeFilter, row.ty);
            } else {
                // @ts-expect-error
                return unifyFunctionTypes(
                    [row],
                    [elem],
                    whereClause,
                    mgens,
                    () => true,
                    unboxingDepth,
                );
            }
        };

        /**
         * Check a query solution for conflicting generics.
         */
        // @ts-expect-error
        const checkTypeMgensForConflict = mgens => {
            if (!mgens) {
                return true;
            }
            const fnTypes = new Set();
            for (const [_qid, fid] of mgens) {
                if (fnTypes.has(fid)) {
                    return false;
                }
                fnTypes.add(fid);
            }
            return true;
        };

        /**
         * Compute an "edit distance" that ignores missing path elements.
         * @param {string[]} contains search query path
         * @param {rustdoc.Row} ty indexed item
         * @returns {null|number} edit distance
         */
        function checkPath(contains, ty) {
            if (contains.length === 0) {
                return 0;
            }
            const maxPathEditDistance = Math.floor(
                contains.reduce((acc, next) => acc + next.length, 0) / 3,
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

        // @ts-expect-error
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

        // this does not yet have a type in `rustdoc.d.ts`.
        // @ts-expect-error
        function createAliasFromItem(item) {
            return {
                crate: item.crate,
                name: item.name,
                path: item.path,
                descShard: item.descShard,
                descIndex: item.descIndex,
                exactPath: item.exactPath,
                ty: item.ty,
                parent: item.parent,
                type: item.type,
                is_alias: true,
                bitIndex: item.bitIndex,
                implDisambiguator: item.implDisambiguator,
            };
        }

        // @ts-expect-error
        const handleAliases = async(ret, query, filterCrates, currentCrate) => {
            const lowerQuery = query.toLowerCase();
            // We separate aliases and crate aliases because we want to have current crate
            // aliases to be before the others in the displayed results.
            // @ts-expect-error
            const aliases = [];
            // @ts-expect-error
            const crateAliases = [];
            if (filterCrates !== null) {
                if (this.ALIASES.has(filterCrates)
                    && this.ALIASES.get(filterCrates).has(lowerQuery)) {
                    const query_aliases = this.ALIASES.get(filterCrates).get(lowerQuery);
                    for (const alias of query_aliases) {
                        aliases.push(createAliasFromItem(this.searchIndex[alias]));
                    }
                }
            } else {
                for (const [crate, crateAliasesIndex] of this.ALIASES) {
                    if (crateAliasesIndex.has(lowerQuery)) {
                        // @ts-expect-error
                        const pushTo = crate === currentCrate ? crateAliases : aliases;
                        const query_aliases = crateAliasesIndex.get(lowerQuery);
                        for (const alias of query_aliases) {
                            pushTo.push(createAliasFromItem(this.searchIndex[alias]));
                        }
                    }
                }
            }

            // @ts-expect-error
            const sortFunc = (aaa, bbb) => {
                if (aaa.path < bbb.path) {
                    return 1;
                } else if (aaa.path === bbb.path) {
                    return 0;
                }
                return -1;
            };
            // @ts-expect-error
            crateAliases.sort(sortFunc);
            aliases.sort(sortFunc);

            // @ts-expect-error
            const fetchDesc = alias => {
                // @ts-expect-error
                return this.searchIndexEmptyDesc.get(alias.crate).contains(alias.bitIndex) ?
                    "" : this.searchState.loadDesc(alias);
            };
            const [crateDescs, descs] = await Promise.all([
                // @ts-expect-error
                Promise.all(crateAliases.map(fetchDesc)),
                Promise.all(aliases.map(fetchDesc)),
            ]);

            // @ts-expect-error
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

            aliases.forEach((alias, i) => {
                // @ts-expect-error
                alias.desc = descs[i];
            });
            aliases.forEach(pushFunc);
            // @ts-expect-error
            crateAliases.forEach((alias, i) => {
                alias.desc = crateDescs[i];
            });
            // @ts-expect-error
            crateAliases.forEach(pushFunc);
        };

        /**
         * This function adds the given result into the provided `results` map if it matches the
         * following condition:
         *
         * * If it is a "literal search" (`parsedQuery.literalSearch`), then `dist` must be 0.
         * * If it is not a "literal search", `dist` must be <= `maxEditDistance`.
         *
         * The `results` map contains information which will be used to sort the search results:
         *
         * * `fullId` is an `integer`` used as the key of the object we use for the `results` map.
         * * `id` is the index in the `searchIndex` array for this element.
         * * `index` is an `integer`` used to sort by the position of the word in the item's name.
         * * `dist` is the main metric used to sort the search results.
         * * `path_dist` is zero if a single-component search query is used, otherwise it's the
         *   distance computed for everything other than the last path component.
         *
         * @param {rustdoc.Results} results
         * @param {number} fullId
         * @param {number} id
         * @param {number} index
         * @param {number} dist
         * @param {number} path_dist
         * @param {number} maxEditDistance
         */
        function addIntoResults(results, fullId, id, index, dist, path_dist, maxEditDistance) {
            if (dist <= maxEditDistance || index !== -1) {
                if (results.has(fullId)) {
                    const result = results.get(fullId);
                    if (result === undefined || result.dontValidate || result.dist <= dist) {
                        return;
                    }
                }
                // @ts-expect-error
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
         * This function is called in case the query has more than one element. In this case, it'll
         * try to match the items which validates all the elements. For `aa -> bb` will look for
         * functions which have a parameter `aa` and has `bb` in its returned values.
         *
         * @param {rustdoc.Row} row
         * @param {number} pos      - Position in the `searchIndex`.
         * @param {rustdoc.Results} results
         */
        function handleArgs(row, pos, results) {
            if (!row || (filterCrates !== null && row.crate !== filterCrates)) {
                return;
            }
            const rowType = row.type;
            if (!rowType) {
                return;
            }

            const tfpDist = compareTypeFingerprints(
                row.id,
                parsedQuery.typeFingerprint,
            );
            if (tfpDist === null) {
                return;
            }
            // @ts-expect-error
            if (results.size >= MAX_RESULTS && tfpDist > results.max_dist) {
                return;
            }

            // If the result is too "bad", we return false and it ends this search.
            if (!unifyFunctionTypes(
                rowType.inputs,
                parsedQuery.elems,
                rowType.where_clause,
                null,
                // @ts-expect-error
                mgens => {
                    return unifyFunctionTypes(
                        rowType.output,
                        parsedQuery.returned,
                        rowType.where_clause,
                        mgens,
                        checkTypeMgensForConflict,
                        0, // unboxing depth
                    );
                },
                0, // unboxing depth
            )) {
                return;
            }

            results.max_dist = Math.max(results.max_dist || 0, tfpDist);
            addIntoResults(results, row.id, pos, 0, tfpDist, 0, Number.MAX_VALUE);
        }

        /**
         * Compare the query fingerprint with the function fingerprint.
         *
         * @param {number} fullId - The function
         * @param {Uint32Array} queryFingerprint - The query
         * @returns {number|null} - Null if non-match, number if distance
         *                          This function might return 0!
         */
        const compareTypeFingerprints = (fullId, queryFingerprint) => {
            const fh0 = this.functionTypeFingerprint[fullId * 4];
            const fh1 = this.functionTypeFingerprint[(fullId * 4) + 1];
            const fh2 = this.functionTypeFingerprint[(fullId * 4) + 2];
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
            return this.functionTypeFingerprint[(fullId * 4) + 3];
        };


        const innerRunQuery = () => {
            if (parsedQuery.foundElems === 1 && !parsedQuery.hasReturnArrow) {
                const elem = parsedQuery.elems[0];
                // use arrow functions to preserve `this`.
                /** @type {function(number): void} */
                const handleNameSearch = id => {
                    const row = this.searchIndex[id];
                    if (!typePassesFilter(elem.typeFilter, row.ty) ||
                        (filterCrates !== null && row.crate !== filterCrates)) {
                        return;
                    }

                    let pathDist = 0;
                    if (elem.fullPath.length > 1) {

                        const maybePathDist = checkPath(elem.pathWithoutLast, row);
                        if (maybePathDist === null) {
                            return;
                        }
                        pathDist = maybePathDist;
                    }

                    if (parsedQuery.literalSearch) {
                        if (row.word === elem.pathLast) {
                            addIntoResults(results_others, row.id, id, 0, 0, pathDist, 0);
                        }
                    } else {
                        addIntoResults(
                            results_others,
                            row.id,
                            id,
                            row.normalizedName.indexOf(elem.normalizedPathLast),
                            editDistance(
                                row.normalizedName,
                                elem.normalizedPathLast,
                                maxEditDistance,
                            ),
                            pathDist,
                            maxEditDistance,
                        );
                    }
                };
                if (elem.normalizedPathLast !== "") {
                    const last = elem.normalizedPathLast;
                    for (const id of this.nameTrie.search(last, this.tailTable)) {
                        handleNameSearch(id);
                    }
                }
                const length = this.searchIndex.length;

                for (let i = 0, nSearchIndex = length; i < nSearchIndex; ++i) {
                    // queries that end in :: bypass the trie
                    if (elem.normalizedPathLast === "") {
                        handleNameSearch(i);
                    }
                    const row = this.searchIndex[i];
                    if (filterCrates !== null && row.crate !== filterCrates) {
                        continue;
                    }
                    const tfpDist = compareTypeFingerprints(
                        row.id,
                        parsedQuery.typeFingerprint,
                    );
                    if (tfpDist !== null) {
                        const in_args = row.type && row.type.inputs
                            && checkIfInList(row.type.inputs, elem, row.type.where_clause, null, 0);
                        const returned = row.type && row.type.output
                            && checkIfInList(row.type.output, elem, row.type.where_clause, null, 0);
                        if (in_args) {
                            results_in_args.max_dist = Math.max(
                                results_in_args.max_dist || 0,
                                tfpDist,
                            );
                            const maxDist = results_in_args.size < MAX_RESULTS ?
                                (tfpDist + 1) :
                                results_in_args.max_dist;
                            addIntoResults(results_in_args, row.id, i, -1, tfpDist, 0, maxDist);
                        }
                        if (returned) {
                            results_returned.max_dist = Math.max(
                                results_returned.max_dist || 0,
                                tfpDist,
                            );
                            const maxDist = results_returned.size < MAX_RESULTS ?
                                (tfpDist + 1) :
                                results_returned.max_dist;
                            addIntoResults(results_returned, row.id, i, -1, tfpDist, 0, maxDist);
                        }
                    }
                }
            } else if (parsedQuery.foundElems > 0) {
                // Sort input and output so that generic type variables go first and
                // types with generic parameters go last.
                // That's because of the way unification is structured: it eats off
                // the end, and hits a fast path if the last item is a simple atom.
                /** @type {function(rustdoc.QueryElement, rustdoc.QueryElement): number} */
                const sortQ = (a, b) => {
                    const ag = a.generics.length === 0 && a.bindings.size === 0;
                    const bg = b.generics.length === 0 && b.bindings.size === 0;
                    if (ag !== bg) {
                        // unary `+` converts booleans into integers.
                        return +ag - +bg;
                    }
                    const ai = a.id !== null && a.id > 0;
                    const bi = b.id !== null && b.id > 0;
                    return +ai - +bi;
                };
                parsedQuery.elems.sort(sortQ);
                parsedQuery.returned.sort(sortQ);
                for (let i = 0, nSearchIndex = this.searchIndex.length; i < nSearchIndex; ++i) {
                    handleArgs(this.searchIndex[i], i, results_others);
                }
            }
        };

        if (parsedQuery.error === null) {
            innerRunQuery();
        }

        const isType = parsedQuery.foundElems !== 1 || parsedQuery.hasReturnArrow;
        const [sorted_in_args, sorted_returned, sorted_others] = await Promise.all([
            sortResults(results_in_args, "elems", currentCrate),
            sortResults(results_returned, "returned", currentCrate),
            // @ts-expect-error
            sortResults(results_others, (isType ? "query" : null), currentCrate),
        ]);
        const ret = createQueryResults(
            sorted_in_args,
            sorted_returned,
            sorted_others,
            parsedQuery);
        await handleAliases(ret, parsedQuery.userQuery.replace(/"/g, ""),
            filterCrates, currentCrate);
        await Promise.all([ret.others, ret.returned, ret.in_args].map(async list => {
            const descs = await Promise.all(list.map(result => {
                // @ts-expect-error
                return this.searchIndexEmptyDesc.get(result.crate).contains(result.bitIndex) ?
                    "" :
                    // @ts-expect-error
                    this.searchState.loadDesc(result);
            }));
            for (const [i, result] of list.entries()) {
                // @ts-expect-error
                result.desc = descs[i];
            }
        }));
        if (parsedQuery.error !== null && ret.others.length !== 0) {
            // It means some doc aliases were found so let's "remove" the error!
            ret.query.error = null;
        }
        return ret;
    }
}


// ==================== Core search logic end ====================

/** @type {Map<string, rustdoc.RawSearchIndexCrate>} */
let rawSearchIndex;
// @ts-expect-error
let docSearch;
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
// @ts-expect-error
let currentResults;

// In the search display, allows to switch between tabs.
// @ts-expect-error
function printTab(nb) {
    let iter = 0;
    let foundCurrentTab = false;
    let foundCurrentResultSet = false;
    // @ts-expect-error
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
    // @ts-expect-error
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
        // @ts-expect-error
        elem.value !== "all crates" &&
        // @ts-expect-error
        window.searchIndex.has(elem.value)
    ) {
        // @ts-expect-error
        return elem.value;
    }
    return null;
}

// @ts-expect-error
function nextTab(direction) {
    const next = (searchState.currentTab + direction + 3) % searchState.focusedByTab.length;
    // @ts-expect-error
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
        // @ts-expect-error
        target.focus();
    }
}

/**
 * Render a set of search results for a single tab.
 * @param {Array<?>}    array   - The search results for this tab
 * @param {rustdoc.ParsedQuery<rustdoc.QueryElement>} query
 * @param {boolean}     display - True if this is the active tab
 */
async function addTab(array, query, display) {
    const extraClass = display ? " active" : "";

    const output = document.createElement(
        array.length === 0 && query.error === null ? "div" : "ul",
    );
    if (array.length > 0) {
        output.className = "search-results " + extraClass;

        const lis = Promise.all(array.map(async item => {
            const name = item.name;
            const type = itemTypes[item.ty];
            const longType = longItemTypes[item.ty];
            const typeName = longType.length !== 0 ? `${longType}` : "?";

            const link = document.createElement("a");
            link.className = "result-" + type;
            link.href = item.href;

            const resultName = document.createElement("span");
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
            if (item.displayTypeSignature) {
                const {type, mappedNames, whereClause} = await item.displayTypeSignature;
                const displayType = document.createElement("div");
                // @ts-expect-error
                type.forEach((value, index) => {
                    if (index % 2 !== 0) {
                        const highlight = document.createElement("strong");
                        highlight.appendChild(document.createTextNode(value));
                        displayType.appendChild(highlight);
                    } else {
                        displayType.appendChild(document.createTextNode(value));
                    }
                });
                if (mappedNames.size > 0 || whereClause.size > 0) {
                    let addWhereLineFn = () => {
                        const line = document.createElement("div");
                        line.className = "where";
                        line.appendChild(document.createTextNode("where"));
                        displayType.appendChild(line);
                        addWhereLineFn = () => {};
                    };
                    for (const [qname, name] of mappedNames) {
                        // don't care unless the generic name is different
                        if (name === qname) {
                            continue;
                        }
                        addWhereLineFn();
                        const line = document.createElement("div");
                        line.className = "where";
                        line.appendChild(document.createTextNode(`    ${qname} matches `));
                        const lineStrong = document.createElement("strong");
                        lineStrong.appendChild(document.createTextNode(name));
                        line.appendChild(lineStrong);
                        displayType.appendChild(line);
                    }
                    for (const [name, innerType] of whereClause) {
                        // don't care unless there's at least one highlighted entry
                        if (innerType.length <= 1) {
                            continue;
                        }
                        addWhereLineFn();
                        const line = document.createElement("div");
                        line.className = "where";
                        line.appendChild(document.createTextNode(`    ${name}: `));
                        // @ts-expect-error
                        innerType.forEach((value, index) => {
                            if (index % 2 !== 0) {
                                const highlight = document.createElement("strong");
                                highlight.appendChild(document.createTextNode(value));
                                line.appendChild(highlight);
                            } else {
                                line.appendChild(document.createTextNode(value));
                            }
                        });
                        displayType.appendChild(line);
                    }
                }
                displayType.className = "type-signature";
                link.appendChild(displayType);
            }

            link.appendChild(description);
            return link;
        }));
        lis.then(lis => {
            for (const li of lis) {
                output.appendChild(li);
            }
        });
    } else if (query.error === null) {
        const dlroChannel = `https://doc.rust-lang.org/${getVar("channel")}`;
        output.className = "search-failed" + extraClass;
        output.innerHTML = "No results :(<br/>" +
            "Try on <a href=\"https://duckduckgo.com/?q=" +
            encodeURIComponent("rust " + query.userQuery) +
            "\">DuckDuckGo</a>?<br/><br/>" +
            "Or try looking in one of these:<ul><li>The <a " +
            `href="${dlroChannel}/reference/index.html">Rust Reference</a> ` +
            " for technical details about the language.</li><li><a " +
            `href="${dlroChannel}/rust-by-example/index.html">Rust By ` +
            "Example</a> for expository code examples.</a></li><li>The <a " +
            `href="${dlroChannel}/book/index.html">Rust Book</a> for ` +
            "introductions to language features and the language itself.</li><li><a " +
            "href=\"https://docs.rs\">Docs.rs</a> for documentation of crates released on" +
            " <a href=\"https://crates.io/\">crates.io</a>.</li></ul>";
    }
    return output;
}

// @ts-expect-error
function makeTabHeader(tabNb, text, nbElems) {
    // https://blog.horizon-eda.org/misc/2020/02/19/ui.html
    //
    // CSS runs with `font-variant-numeric: tabular-nums` to ensure all
    // digits are the same width. \u{2007} is a Unicode space character
    // that is defined to be the same width as a digit.
    const fmtNbElems =
        nbElems < 10  ? `\u{2007}(${nbElems})\u{2007}\u{2007}` :
        nbElems < 100 ? `\u{2007}(${nbElems})\u{2007}` : `\u{2007}(${nbElems})`;
    if (searchState.currentTab === tabNb) {
        return "<button class=\"selected\">" + text +
            "<span class=\"count\">" + fmtNbElems + "</span></button>";
    }
    return "<button>" + text + "<span class=\"count\">" + fmtNbElems + "</span></button>";
}

/**
 * @param {rustdoc.ResultsTable} results
 * @param {boolean} go_to_first
 * @param {string} filterCrates
 */
async function showResults(results, go_to_first, filterCrates) {
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
        window.onunload = () => { };
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
        // @ts-expect-error
        results.query = DocSearch.parseQuery(searchState.input.value);
    }

    currentResults = results.query.userQuery;

    // Navigate to the relevant tab if the current tab is empty, like in case users search
    // for "-> String". If they had selected another tab previously, they have to click on
    // it again.
    let currentTab = searchState.currentTab;
    if ((currentTab === 0 && results.others.length === 0) ||
        (currentTab === 1 && results.in_args.length === 0) ||
        (currentTab === 2 && results.returned.length === 0)) {
        if (results.others.length !== 0) {
            currentTab = 0;
        } else if (results.in_args.length) {
            currentTab = 1;
        } else if (results.returned.length) {
            currentTab = 2;
        }
    }

    let crates = "";
    if (rawSearchIndex.size > 1) {
        crates = "<div class=\"sub-heading\"> in&nbsp;<div id=\"crate-search-div\">" +
            "<select id=\"crate-search\"><option value=\"all crates\">all crates</option>";
        for (const c of rawSearchIndex.keys()) {
            crates += `<option value="${c}" ${c === filterCrates && "selected"}>${c}</option>`;
        }
        crates += "</select></div></div>";
    }

    let output = `<div class="main-heading">\
        <h1 class="search-results-title">Results</h1>${crates}</div>`;
    if (results.query.error !== null) {
        const error = results.query.error;
        // @ts-expect-error
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
            makeTabHeader(0, "In Names", results.others.length) +
            "</div>";
        currentTab = 0;
    } else if (results.query.foundElems <= 1 && results.query.returned.length === 0) {
        output += "<div id=\"search-tabs\">" +
            makeTabHeader(0, "In Names", results.others.length) +
            makeTabHeader(1, "In Parameters", results.in_args.length) +
            makeTabHeader(2, "In Return Types", results.returned.length) +
            "</div>";
    } else {
        const signatureTabTitle =
            results.query.elems.length === 0 ? "In Function Return Types" :
                results.query.returned.length === 0 ? "In Function Parameters" :
                    "In Function Signatures";
        output += "<div id=\"search-tabs\">" +
            makeTabHeader(0, signatureTabTitle, results.others.length) +
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

    const [ret_others, ret_in_args, ret_returned] = await Promise.all([
        addTab(results.others, results.query, currentTab === 0),
        addTab(results.in_args, results.query, currentTab === 1),
        addTab(results.returned, results.query, currentTab === 2),
    ]);

    const resultsElem = document.createElement("div");
    resultsElem.id = "results";
    resultsElem.appendChild(ret_others);
    resultsElem.appendChild(ret_in_args);
    resultsElem.appendChild(ret_returned);

    // @ts-expect-error
    search.innerHTML = output;
    if (searchState.rustdocToolbar) {
        // @ts-expect-error
        search.querySelector(".main-heading").appendChild(searchState.rustdocToolbar);
    }
    const crateSearch = document.getElementById("crate-search");
    if (crateSearch) {
        crateSearch.addEventListener("input", updateCrate);
    }
    // @ts-expect-error
    search.appendChild(resultsElem);
    // Reset focused elements.
    searchState.showResults(search);
    // @ts-expect-error
    const elems = document.getElementById("search-tabs").childNodes;
    // @ts-expect-error
    searchState.focusedByTab = [];
    let i = 0;
    for (const elem of elems) {
        const j = i;
        // @ts-expect-error
        elem.onclick = () => printTab(j);
        searchState.focusedByTab.push(null);
        i += 1;
    }
    printTab(currentTab);
}

// @ts-expect-error
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
async function search(forced) {
    // @ts-expect-error
    const query = DocSearch.parseQuery(searchState.input.value.trim());
    let filterCrates = getFilterCrates();

    // @ts-expect-error
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
    searchState.title = "\"" + query.userQuery + "\" Search - Rust";

    // Because searching is incremental by character, only the most
    // recent search query is added to the browser history.
    updateSearchHistory(buildUrl(query.userQuery, filterCrates));

    await showResults(
        // @ts-expect-error
        await docSearch.execQuery(query, filterCrates, window.currentCrate),
        params.go_to_first,
        // @ts-expect-error
        filterCrates);
}

/**
 * Callback for when the search form is submitted.
 * @param {Event} [e] - The event that triggered this call, if any
 */
function onSearchSubmit(e) {
    // @ts-expect-error
    e.preventDefault();
    searchState.clearInputTimeout();
    search();
}

function putBackSearch() {
    const search_input = searchState.input;
    if (!searchState.input) {
        return;
    }
    // @ts-expect-error
    if (search_input.value !== "" && !searchState.isDisplayed()) {
        searchState.showResults();
        if (browserSupportsHistoryApi()) {
            history.replaceState(null, "",
                // @ts-expect-error
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
    // @ts-expect-error
    if (searchState.input.value === "") {
        // @ts-expect-error
        searchState.input.value = params.search || "";
    }

    const searchAfter500ms = () => {
        searchState.clearInputTimeout();
        // @ts-expect-error
        if (searchState.input.value.length === 0) {
            searchState.hideResults();
        } else {
            searchState.timeout = setTimeout(search, 500);
        }
    };
    // @ts-expect-error
    searchState.input.onkeyup = searchAfter500ms;
    // @ts-expect-error
    searchState.input.oninput = searchAfter500ms;
    // @ts-expect-error
    document.getElementsByClassName("search-form")[0].onsubmit = onSearchSubmit;
    // @ts-expect-error
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
    // @ts-expect-error
    searchState.input.onpaste = searchState.input.onchange;

    // @ts-expect-error
    searchState.outputElement().addEventListener("keydown", e => {
        // We only handle unmodified keystrokes here. We don't want to interfere with,
        // for instance, alt-left and alt-right for history navigation.
        if (e.altKey || e.ctrlKey || e.shiftKey || e.metaKey) {
            return;
        }
        // up and down arrow select next/previous search result, or the
        // search box if we're already at the top.
        if (e.which === 38) { // up
            // @ts-expect-error
            const previous = document.activeElement.previousElementSibling;
            if (previous) {
                // @ts-expect-error
                previous.focus();
            } else {
                searchState.focus();
            }
            e.preventDefault();
        } else if (e.which === 40) { // down
            // @ts-expect-error
            const next = document.activeElement.nextElementSibling;
            if (next) {
                // @ts-expect-error
                next.focus();
            }
            // @ts-expect-error
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

    // @ts-expect-error
    searchState.input.addEventListener("keydown", e => {
        if (e.which === 40) { // down
            focusSearchResult();
            e.preventDefault();
        }
    });

    // @ts-expect-error
    searchState.input.addEventListener("focus", () => {
        putBackSearch();
    });

    // @ts-expect-error
    searchState.input.addEventListener("blur", () => {
        if (window.searchState.input) {
            window.searchState.input.placeholder = window.searchState.origPlaceholder;
        }
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
                // @ts-expect-error
                searchState.input.value = params.search;
                // Some browsers fire "onpopstate" for every page load
                // (Chrome), while others fire the event only when actually
                // popping a state (Firefox), which is why search() is
                // called both here and at the end of the startSearch()
                // function.
                e.preventDefault();
                search();
            } else {
                // @ts-expect-error
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
        // @ts-expect-error
        if (searchState.input.value === "" && qSearch) {
            // @ts-expect-error
            searchState.input.value = qSearch;
        }
        search();
    };
}

// @ts-expect-error
function updateCrate(ev) {
    if (ev.target.value === "all crates") {
        // If we don't remove it from the URL, it'll be picked up again by the search.
        // @ts-expect-error
        const query = searchState.input.value.trim();
        updateSearchHistory(buildUrl(query, null));
    }
    // In case you "cut" the entry from the search input, then change the crate filter
    // before paste back the previous search, you get the old search results without
    // the filter. To prevent this, we need to remove the previous results.
    currentResults = null;
    search(true);
}

// Parts of this code are based on Lucene, which is licensed under the
// Apache/2.0 license.
// More information found here:
// https://fossies.org/linux/lucene/lucene/core/src/java/org/apache/lucene/util/automaton/
//   LevenshteinAutomata.java
class ParametricDescription {
    // @ts-expect-error
    constructor(w, n, minErrors) {
        this.w = w;
        this.n = n;
        this.minErrors = minErrors;
    }
    // @ts-expect-error
    isAccept(absState) {
        const state = Math.floor(absState / (this.w + 1));
        const offset = absState % (this.w + 1);
        return this.w - offset + this.minErrors[state] <= this.n;
    }
    // @ts-expect-error
    getPosition(absState) {
        return absState % (this.w + 1);
    }
    // @ts-expect-error
    getVector(name, charCode, pos, end) {
        let vector = 0;
        for (let i = pos; i < end; i += 1) {
            vector = vector << 1;
            if (name.charCodeAt(i) === charCode) {
                vector |= 1;
            }
        }
        return vector;
    }
    // @ts-expect-error
    unpack(data, index, bitsPerValue) {
        const bitLoc = (bitsPerValue * index);
        const dataLoc = bitLoc >> 5;
        const bitStart = bitLoc & 31;
        if (bitStart + bitsPerValue <= 32) {
            // not split
            return ((data[dataLoc] >> bitStart) & this.MASKS[bitsPerValue - 1]);
        } else {
            // split
            const part = 32 - bitStart;
            return ~~(((data[dataLoc] >> bitStart) & this.MASKS[part - 1]) +
                ((data[1 + dataLoc] & this.MASKS[bitsPerValue - part - 1]) << part));
        }
    }
}
ParametricDescription.prototype.MASKS = new Int32Array([
    0x1, 0x3, 0x7, 0xF,
    0x1F, 0x3F, 0x7F, 0xFF,
    0x1FF, 0x3F, 0x7FF, 0xFFF,
    0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF,
    0x1FFFF, 0x3FFFF, 0x7FFFF, 0xFFFFF,
    0x1FFFFF, 0x3FFFFF, 0x7FFFFF, 0xFFFFFF,
    0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF, 0xFFFFFFF,
    0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF,
]);

// The following code was generated with the moman/finenight pkg
// This package is available under the MIT License, see NOTICE.txt
// for more details.
// This class is auto-generated, Please do not modify it directly.
// You should modify the https://gitlab.com/notriddle/createAutomata.py instead.
// The following code was generated with the moman/finenight pkg
// This package is available under the MIT License, see NOTICE.txt
// for more details.
// This class is auto-generated, Please do not modify it directly.
// You should modify https://gitlab.com/notriddle/moman-rustdoc instead.

class Lev2TParametricDescription extends ParametricDescription {
    /**
     * @param {number} absState
     * @param {number} position
     * @param {number} vector
     * @returns {number}
    */
    transition(absState, position, vector) {
        let state = Math.floor(absState / (this.w + 1));
        let offset = absState % (this.w + 1);

        if (position === this.w) {
            if (state < 3) { // eslint-disable-line no-lonely-if
                const loc = Math.imul(vector, 3) + state;
                offset += this.unpack(this.offsetIncrs0, loc, 1);
                state = this.unpack(this.toStates0, loc, 2) - 1;
            }
        } else if (position === this.w - 1) {
            if (state < 5) { // eslint-disable-line no-lonely-if
                const loc = Math.imul(vector, 5) + state;
                offset += this.unpack(this.offsetIncrs1, loc, 1);
                state = this.unpack(this.toStates1, loc, 3) - 1;
            }
        } else if (position === this.w - 2) {
            if (state < 13) { // eslint-disable-line no-lonely-if
                const loc = Math.imul(vector, 13) + state;
                offset += this.unpack(this.offsetIncrs2, loc, 2);
                state = this.unpack(this.toStates2, loc, 4) - 1;
            }
        } else if (position === this.w - 3) {
            if (state < 28) { // eslint-disable-line no-lonely-if
                const loc = Math.imul(vector, 28) + state;
                offset += this.unpack(this.offsetIncrs3, loc, 2);
                state = this.unpack(this.toStates3, loc, 5) - 1;
            }
        } else if (position === this.w - 4) {
            if (state < 45) { // eslint-disable-line no-lonely-if
                const loc = Math.imul(vector, 45) + state;
                offset += this.unpack(this.offsetIncrs4, loc, 3);
                state = this.unpack(this.toStates4, loc, 6) - 1;
            }
        } else {
            if (state < 45) { // eslint-disable-line no-lonely-if
                const loc = Math.imul(vector, 45) + state;
                offset += this.unpack(this.offsetIncrs5, loc, 3);
                state = this.unpack(this.toStates5, loc, 6) - 1;
            }
        }

        if (state === -1) {
            // null state
            return -1;
        } else {
            // translate back to abs
            return Math.imul(state, this.w + 1) + offset;
        }
    }

    // state map
    //   0 -> [(0, 0)]
    //   1 -> [(0, 1)]
    //   2 -> [(0, 2)]
    //   3 -> [(0, 1), (1, 1)]
    //   4 -> [(0, 2), (1, 2)]
    //   5 -> [(0, 1), (1, 1), (2, 1)]
    //   6 -> [(0, 2), (1, 2), (2, 2)]
    //   7 -> [(0, 1), (2, 1)]
    //   8 -> [(0, 1), (2, 2)]
    //   9 -> [(0, 2), (2, 1)]
    //   10 -> [(0, 2), (2, 2)]
    //   11 -> [t(0, 1), (0, 1), (1, 1), (2, 1)]
    //   12 -> [t(0, 2), (0, 2), (1, 2), (2, 2)]
    //   13 -> [(0, 2), (1, 2), (2, 2), (3, 2)]
    //   14 -> [(0, 1), (1, 1), (3, 2)]
    //   15 -> [(0, 1), (2, 2), (3, 2)]
    //   16 -> [(0, 1), (3, 2)]
    //   17 -> [(0, 1), t(1, 2), (2, 2), (3, 2)]
    //   18 -> [(0, 2), (1, 2), (3, 1)]
    //   19 -> [(0, 2), (1, 2), (3, 2)]
    //   20 -> [(0, 2), (1, 2), t(1, 2), (2, 2), (3, 2)]
    //   21 -> [(0, 2), (2, 1), (3, 1)]
    //   22 -> [(0, 2), (2, 2), (3, 2)]
    //   23 -> [(0, 2), (3, 1)]
    //   24 -> [(0, 2), (3, 2)]
    //   25 -> [(0, 2), t(1, 2), (1, 2), (2, 2), (3, 2)]
    //   26 -> [t(0, 2), (0, 2), (1, 2), (2, 2), (3, 2)]
    //   27 -> [t(0, 2), (0, 2), (1, 2), (3, 1)]
    //   28 -> [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
    //   29 -> [(0, 2), (1, 2), (2, 2), (4, 2)]
    //   30 -> [(0, 2), (1, 2), (2, 2), t(2, 2), (3, 2), (4, 2)]
    //   31 -> [(0, 2), (1, 2), (3, 2), (4, 2)]
    //   32 -> [(0, 2), (1, 2), (4, 2)]
    //   33 -> [(0, 2), (1, 2), t(1, 2), (2, 2), (3, 2), (4, 2)]
    //   34 -> [(0, 2), (1, 2), t(2, 2), (2, 2), (3, 2), (4, 2)]
    //   35 -> [(0, 2), (2, 1), (4, 2)]
    //   36 -> [(0, 2), (2, 2), (3, 2), (4, 2)]
    //   37 -> [(0, 2), (2, 2), (4, 2)]
    //   38 -> [(0, 2), (3, 2), (4, 2)]
    //   39 -> [(0, 2), (4, 2)]
    //   40 -> [(0, 2), t(1, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
    //   41 -> [(0, 2), t(2, 2), (2, 2), (3, 2), (4, 2)]
    //   42 -> [t(0, 2), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
    //   43 -> [t(0, 2), (0, 2), (1, 2), (2, 2), (4, 2)]
    //   44 -> [t(0, 2), (0, 2), (1, 2), (2, 2), t(2, 2), (3, 2), (4, 2)]


    /** @param {number} w - length of word being checked */
    constructor(w) {
        super(w, 2, new Int32Array([
            0,1,2,0,1,-1,0,-1,0,-1,0,-1,0,-1,-1,-1,-1,-1,-2,-1,-1,-2,-1,-2,
            -1,-1,-1,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,
        ]));
    }
}

Lev2TParametricDescription.prototype.toStates0 = /*2 bits per value */ new Int32Array([
    0xe,
]);
Lev2TParametricDescription.prototype.offsetIncrs0 = /*1 bits per value */ new Int32Array([
    0x0,
]);

Lev2TParametricDescription.prototype.toStates1 = /*3 bits per value */ new Int32Array([
    0x1a688a2c,
]);
Lev2TParametricDescription.prototype.offsetIncrs1 = /*1 bits per value */ new Int32Array([
    0x3e0,
]);

Lev2TParametricDescription.prototype.toStates2 = /*4 bits per value */ new Int32Array([
    0x70707054,0xdc07035,0x3dd3a3a,0x2323213a,
    0x15435223,0x22545432,0x5435,
]);
Lev2TParametricDescription.prototype.offsetIncrs2 = /*2 bits per value */ new Int32Array([
    0x80000,0x55582088,0x55555555,0x55,
]);

Lev2TParametricDescription.prototype.toStates3 = /*5 bits per value */ new Int32Array([
    0x1c0380a4,0x700a570,0xca529c0,0x180a00,
    0xa80af180,0xc5498e60,0x5a546398,0x8c4300e8,
    0xac18c601,0xd8d43501,0x863500ad,0x51976d6a,
    0x8ca0180a,0xc3501ac2,0xb0c5be16,0x76dda8a5,
    0x18c4519,0xc41294a,0xe248d231,0x1086520c,
    0xce31ac42,0x13946358,0x2d0348c4,0x6732d494,
    0x1ad224a5,0xd635ad4b,0x520c4139,0xce24948,
    0x22110a52,0x58ce729d,0xc41394e3,0x941cc520,
    0x90e732d4,0x4729d224,0x39ce35ad,
]);
Lev2TParametricDescription.prototype.offsetIncrs3 = /*2 bits per value */ new Int32Array([
    0x80000,0xc0c830,0x300f3c30,0x2200fcff,
    0xcaa00a08,0x3c2200a8,0xa8fea00a,0x55555555,
    0x55555555,0x55555555,0x55555555,0x55555555,
    0x55555555,0x55555555,
]);

Lev2TParametricDescription.prototype.toStates4 = /*6 bits per value */ new Int32Array([
    0x801c0144,0x1453803,0x14700038,0xc0005145,
    0x1401,0x14,0x140000,0x0,
    0x510000,0x6301f007,0x301f00d1,0xa186178,
    0xc20ca0c3,0xc20c30,0xc30030c,0xc00c00cd,
    0xf0c00c30,0x4c054014,0xc30944c3,0x55150c34,
    0x8300550,0x430c0143,0x50c31,0xc30850c,
    0xc3143000,0x50053c50,0x5130d301,0x850d30c2,
    0x30a08608,0xc214414,0x43142145,0x21450031,
    0x1400c314,0x4c143145,0x32832803,0x28014d6c,
    0xcd34a0c3,0x1c50c76,0x1c314014,0x430c30c3,
    0x1431,0xc300500,0xca00d303,0xd36d0e40,
    0x90b0e400,0xcb2abb2c,0x70c20ca1,0x2c32ca2c,
    0xcd2c70cb,0x31c00c00,0x34c2c32c,0x5583280,
    0x558309b7,0x6cd6ca14,0x430850c7,0x51c51401,
    0x1430c714,0xc3087,0x71451450,0xca00d30,
    0xc26dc156,0xb9071560,0x1cb2abb2,0xc70c2144,
    0xb1c51ca1,0x1421c70c,0xc51c00c3,0x30811c51,
    0x24324308,0xc51031c2,0x70820820,0x5c33830d,
    0xc33850c3,0x30c30c30,0xc30c31c,0x451450c3,
    0x20c20c20,0xda0920d,0x5145914f,0x36596114,
    0x51965865,0xd9643653,0x365a6590,0x51964364,
    0x43081505,0x920b2032,0x2c718b28,0xd7242249,
    0x35cb28b0,0x2cb3872c,0x972c30d7,0xb0c32cb2,
    0x4e1c75c,0xc80c90c2,0x62ca2482,0x4504171c,
    0xd65d9610,0x33976585,0xd95cb5d,0x4b5ca5d7,
    0x73975c36,0x10308138,0xc2245105,0x41451031,
    0x14e24208,0xc35c3387,0x51453851,0x1c51c514,
    0xc70c30c3,0x20451450,0x14f1440c,0x4f0da092,
    0x4513d41,0x6533944d,0x1350e658,0xe1545055,
    0x64365a50,0x5519383,0x51030815,0x28920718,
    0x441c718b,0x714e2422,0x1c35cb28,0x4e1c7387,
    0xb28e1c51,0x5c70c32c,0xc204e1c7,0x81c61440,
    0x1c62ca24,0xd04503ce,0x85d63944,0x39338e65,
    0x8e154387,0x364b5ca3,0x38739738,
]);
Lev2TParametricDescription.prototype.offsetIncrs4 = /*3 bits per value */ new Int32Array([
    0x10000000,0xc00000,0x60061,0x400,
    0x0,0x80010008,0x249248a4,0x8229048,
    0x2092,0x6c3603,0xb61b6c30,0x6db6036d,
    0xdb6c0,0x361b0180,0x91b72000,0xdb11b71b,
    0x6db6236,0x1008200,0x12480012,0x24924906,
    0x48200049,0x80410002,0x24000900,0x4924a489,
    0x10822492,0x20800125,0x48360,0x9241b692,
    0x6da4924,0x40009268,0x241b010,0x291b4900,
    0x6d249249,0x49493423,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x92492492,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x92492492,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x2492,
]);

Lev2TParametricDescription.prototype.toStates5 = /*6 bits per value */ new Int32Array([
    0x801c0144,0x1453803,0x14700038,0xc0005145,
    0x1401,0x14,0x140000,0x0,
    0x510000,0x4e00e007,0xe0051,0x3451451c,
    0xd015000,0x30cd0000,0xc30c30c,0xc30c30d4,
    0x40c30c30,0x7c01c014,0xc03458c0,0x185e0c07,
    0x2830c286,0x830c3083,0xc30030,0x33430c,
    0x30c3003,0x70051030,0x16301f00,0x8301f00d,
    0x30a18617,0xc20ca0c,0x431420c3,0xb1450c51,
    0x14314315,0x4f143145,0x34c05401,0x4c30944c,
    0x55150c3,0x30830055,0x1430c014,0xc00050c3,
    0xc30850,0xc314300,0x150053c5,0x25130d30,
    0x5430d30c,0xc0354154,0x300d0c90,0x1cb2cd0c,
    0xc91cb0c3,0x72c30cb2,0x14f1cb2c,0xc34c0540,
    0x34c30944,0x82182214,0x851050c2,0x50851430,
    0x1400c50c,0x30c5085,0x50c51450,0x150053c,
    0xc25130d3,0x8850d30,0x1430a086,0x450c2144,
    0x51cb1c21,0x1c91c70c,0xc71c314b,0x34c1cb1,
    0x6c328328,0xc328014d,0x76cd34a0,0x1401c50c,
    0xc31c3140,0x31430c30,0x14,0x30c3005,
    0xa0ca00d3,0x535b0c,0x4d2830ca,0x514369b3,
    0xc500d01,0x5965965a,0x30d46546,0x6435030c,
    0x8034c659,0xdb439032,0x2c390034,0xcaaecb24,
    0x30832872,0xcb28b1c,0x4b1c32cb,0x70030033,
    0x30b0cb0c,0xe40ca00d,0x400d36d0,0xb2c90b0e,
    0xca1cb2ab,0xa2c70c20,0x6575d95c,0x4315b5ce,
    0x95c53831,0x28034c5d,0x9b705583,0xa1455830,
    0xc76cd6c,0x40143085,0x71451c51,0x871430c,
    0x450000c3,0xd3071451,0x1560ca00,0x560c26dc,
    0xb35b2851,0xc914369,0x1a14500d,0x46593945,
    0xcb2c939,0x94507503,0x328034c3,0x9b70558,
    0xe41c5583,0x72caaeca,0x1c308510,0xc7147287,
    0x50871c32,0x1470030c,0xd307147,0xc1560ca0,
    0x1560c26d,0xabb2b907,0x21441cb2,0x38a1c70c,
    0x8e657394,0x314b1c93,0x39438738,0x43083081,
    0x31c22432,0x820c510,0x830d7082,0x50c35c33,
    0xc30c338,0xc31c30c3,0x50c30c30,0xc204514,
    0x890c90c2,0x31440c70,0xa8208208,0xea0df0c3,
    0x8a231430,0xa28a28a2,0x28a28a1e,0x1861868a,
    0x48308308,0xc3682483,0x14516453,0x4d965845,
    0xd4659619,0x36590d94,0xd969964,0x546590d9,
    0x20c20541,0x920d20c,0x5914f0da,0x96114514,
    0x65865365,0xe89d3519,0x99e7a279,0x9e89e89e,
    0x81821827,0xb2032430,0x18b28920,0x422492c7,
    0xb28b0d72,0x3872c35c,0xc30d72cb,0x32cb2972,
    0x1c75cb0c,0xc90c204e,0xa2482c80,0x24b1c62c,
    0xc3a89089,0xb0ea2e42,0x9669a31c,0xa4966a28,
    0x59a8a269,0x8175e7a,0xb203243,0x718b2892,
    0x4114105c,0x17597658,0x74ce5d96,0x5c36572d,
    0xd92d7297,0xe1ce5d70,0xc90c204,0xca2482c8,
    0x4171c62,0x5d961045,0x976585d6,0x79669533,
    0x964965a2,0x659689e6,0x308175e7,0x24510510,
    0x451031c2,0xe2420841,0x5c338714,0x453851c3,
    0x51c51451,0xc30c31c,0x451450c7,0x41440c20,
    0xc708914,0x82105144,0xf1c58c90,0x1470ea0d,
    0x61861863,0x8a1e85e8,0x8687a8a2,0x3081861,
    0x24853c51,0x5053c368,0x1341144f,0x96194ce5,
    0x1544d439,0x94385514,0xe0d90d96,0x5415464,
    0x4f1440c2,0xf0da0921,0x4513d414,0x533944d0,
    0x350e6586,0x86082181,0xe89e981d,0x18277689,
    0x10308182,0x89207185,0x41c718b2,0x14e24224,
    0xc35cb287,0xe1c73871,0x28e1c514,0xc70c32cb,
    0x204e1c75,0x1c61440c,0xc62ca248,0x90891071,
    0x2e41c58c,0xa31c70ea,0xe86175e7,0xa269a475,
    0x5e7a57a8,0x51030817,0x28920718,0xf38718b,
    0xe5134114,0x39961758,0xe1ce4ce,0x728e3855,
    0x5ce0d92d,0xc204e1ce,0x81c61440,0x1c62ca24,
    0xd04503ce,0x85d63944,0x75338e65,0x5d86075e,
    0x89e69647,0x75e76576,
]);
Lev2TParametricDescription.prototype.offsetIncrs5 = /*3 bits per value */ new Int32Array([
    0x10000000,0xc00000,0x60061,0x400,
    0x0,0x60000008,0x6b003080,0xdb6ab6db,
    0x2db6,0x800400,0x49245240,0x11482412,
    0x104904,0x40020000,0x92292000,0xa4b25924,
    0x9649658,0xd80c000,0xdb0c001b,0x80db6d86,
    0x6db01b6d,0xc0600003,0x86000d86,0x6db6c36d,
    0xddadb6ed,0x300001b6,0x6c360,0xe37236e4,
    0x46db6236,0xdb6c,0x361b018,0xb91b7200,
    0x6dbb1b71,0x6db763,0x20100820,0x61248001,
    0x92492490,0x24820004,0x8041000,0x92400090,
    0x24924830,0x555b6a49,0x2080012,0x20004804,
    0x49252449,0x84112492,0x4000928,0x240201,
    0x92922490,0x58924924,0x49456,0x120d8082,
    0x6da4800,0x69249249,0x249a01b,0x6c04100,
    0x6d240009,0x92492483,0x24d5adb4,0x60208001,
    0x92000483,0x24925236,0x6846da49,0x10400092,
    0x241b0,0x49291b49,0x636d2492,0x92494935,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x92492492,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x92492492,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x92492492,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x92492492,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x92492492,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,
]);

class Lev1TParametricDescription extends ParametricDescription {
    /**
     * @param {number} absState
     * @param {number} position
     * @param {number} vector
     * @returns {number}
    */
    transition(absState, position, vector) {
        let state = Math.floor(absState / (this.w + 1));
        let offset = absState % (this.w + 1);

        if (position === this.w) {
            if (state < 2) { // eslint-disable-line no-lonely-if
                const loc = Math.imul(vector, 2) + state;
                offset += this.unpack(this.offsetIncrs0, loc, 1);
                state = this.unpack(this.toStates0, loc, 2) - 1;
            }
        } else if (position === this.w - 1) {
            if (state < 3) { // eslint-disable-line no-lonely-if
                const loc = Math.imul(vector, 3) + state;
                offset += this.unpack(this.offsetIncrs1, loc, 1);
                state = this.unpack(this.toStates1, loc, 2) - 1;
            }
        } else if (position === this.w - 2) {
            if (state < 6) { // eslint-disable-line no-lonely-if
                const loc = Math.imul(vector, 6) + state;
                offset += this.unpack(this.offsetIncrs2, loc, 2);
                state = this.unpack(this.toStates2, loc, 3) - 1;
            }
        } else {
            if (state < 6) { // eslint-disable-line no-lonely-if
                const loc = Math.imul(vector, 6) + state;
                offset += this.unpack(this.offsetIncrs3, loc, 2);
                state = this.unpack(this.toStates3, loc, 3) - 1;
            }
        }

        if (state === -1) {
            // null state
            return -1;
        } else {
            // translate back to abs
            return Math.imul(state, this.w + 1) + offset;
        }
    }

    // state map
    //   0 -> [(0, 0)]
    //   1 -> [(0, 1)]
    //   2 -> [(0, 1), (1, 1)]
    //   3 -> [(0, 1), (1, 1), (2, 1)]
    //   4 -> [(0, 1), (2, 1)]
    //   5 -> [t(0, 1), (0, 1), (1, 1), (2, 1)]


    /** @param {number} w - length of word being checked */
    constructor(w) {
        super(w, 1, new Int32Array([0,1,0,-1,-1,-1]));
    }
}

Lev1TParametricDescription.prototype.toStates0 = /*2 bits per value */ new Int32Array([
    0x2,
]);
Lev1TParametricDescription.prototype.offsetIncrs0 = /*1 bits per value */ new Int32Array([
    0x0,
]);

Lev1TParametricDescription.prototype.toStates1 = /*2 bits per value */ new Int32Array([
    0xa43,
]);
Lev1TParametricDescription.prototype.offsetIncrs1 = /*1 bits per value */ new Int32Array([
    0x38,
]);

Lev1TParametricDescription.prototype.toStates2 = /*3 bits per value */ new Int32Array([
    0x12180003,0xb45a4914,0x69,
]);
Lev1TParametricDescription.prototype.offsetIncrs2 = /*2 bits per value */ new Int32Array([
    0x558a0000,0x5555,
]);

Lev1TParametricDescription.prototype.toStates3 = /*3 bits per value */ new Int32Array([
    0x900c0003,0xa1904864,0x45a49169,0x5a6d196a,
    0x9634,
]);
Lev1TParametricDescription.prototype.offsetIncrs3 = /*2 bits per value */ new Int32Array([
    0xa0fc0000,0x5555ba08,0x55555555,
]);

// ====================
// WARNING: Nothing should be added below this comment: we need the `initSearch` function to
// be called ONLY when the whole file has been parsed and loaded.

// @ts-expect-error
function initSearch(searchIndx) {
    rawSearchIndex = searchIndx;
    if (typeof window !== "undefined") {
        // @ts-expect-error
        docSearch = new DocSearch(rawSearchIndex, ROOT_PATH, searchState);
        registerSearchEvents();
        // If there's a search term in the URL, execute the search now.
        if (window.searchState.getQueryStringParams().search) {
            search();
        }
    } else if (typeof exports !== "undefined") {
        // @ts-expect-error
        docSearch = new DocSearch(rawSearchIndex, ROOT_PATH, searchState);
        exports.docSearch = docSearch;
        exports.parseQuery = DocSearch.parseQuery;
    }
}

if (typeof exports !== "undefined") {
    exports.initSearch = initSearch;
}

if (typeof window !== "undefined") {
    // @ts-expect-error
    window.initSearch = initSearch;
    // @ts-expect-error
    if (window.searchIndex !== undefined) {
        // @ts-expect-error
        initSearch(window.searchIndex);
    }
} else {
    // Running in Node, not a browser. Run initSearch just to produce the
    // exports.
    initSearch(new Map());
}
