// ignore-tidy-filelength
/* global addClass, getNakedUrl, getVar, nonnull, getSettingValue */
/* global onEachLazy, removeClass, searchState, browserSupportsHistoryApi */

"use strict";

/**
 * @param {stringdex.Stringdex} Stringdex
 * @param {typeof stringdex.RoaringBitmap} RoaringBitmap
 * @param {stringdex.Hooks} hooks
 */
const initSearch = async function(Stringdex, RoaringBitmap, hooks) {

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
 * @param {function(T): Promise<any>} func
 * @param {function(T): boolean} funcBtwn
 */
async function onEachBtwnAsync(arr, func, funcBtwn) {
    let skipped = true;
    for (const value of arr) {
        if (!skipped) {
            funcBtwn(value);
        }
        skipped = await func(value);
    }
}

/**
 * Allow the browser to redraw.
 * @returns {Promise<void>}
 */
const yieldToBrowser = typeof window !== "undefined" && window.requestIdleCallback ?
    function() {
        return new Promise((resolve, _reject) => {
            window.requestIdleCallback(resolve);
        });
    } :
    function() {
        return new Promise((resolve, _reject) => {
            setTimeout(resolve, 0);
        });
    };

/**
 * Promise-based timer wrapper.
 * @param {number} ms
 * @returns {Promise<void>}
 */
const timeout = function(ms) {
    return new Promise((resolve, _reject) => {
        setTimeout(resolve, ms);
    });
};

if (!Promise.withResolvers) {
    /**
     * Polyfill
     * @template T
     * @returns {{
            "promise": Promise<T>,
            "resolve": (function(T): void),
            "reject": (function(any): void)
        }}
     */
    Promise.withResolvers = () => {
        let resolve, reject;
        const promise = new Promise((res, rej) => {
          resolve = res;
          reject = rej;
        });
        // @ts-expect-error
        return {promise, resolve, reject};
    };
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
    "attribute",
];

// used for special search precedence
/** @type {rustdoc.ItemType} */
const TY_PRIMITIVE = 1;
/** @type {rustdoc.ItemType} */
const TY_GENERIC = 26;
/** @type {rustdoc.ItemType} */
const TY_IMPORT = 4;
/** @type {rustdoc.ItemType} */
const TY_TRAIT = 10;
/** @type {rustdoc.ItemType} */
const TY_FN = 7;
/** @type {rustdoc.ItemType} */
const TY_METHOD = 13;
/** @type {rustdoc.ItemType} */
const TY_TYMETHOD = 12;
/** @type {rustdoc.ItemType} */
const TY_ASSOCTYPE = 17;
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
 * Same thing as ItemType::is_fn_like in item_type.rs
 *
 * @param {rustdoc.ItemType} ty
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

    /** @type {function(string, string): void} */
    const handleRefOrPtr = (chr, name) => {
            if (parserState.typeFilter !== null && parserState.typeFilter !== "primitive") {
            throw [
                "Invalid search type: primitive ",
                chr,
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
        const pos = parserState.pos;
        if (parserState.userQuery.slice(pos, pos + 3) === "mut") {
            generics.push(makePrimitiveElement("mut", { typeFilter: "keyword" }));
            parserState.pos += 3;
            c = parserState.userQuery[parserState.pos];
        } else if (chr === "*" && parserState.userQuery.slice(pos, pos + 5) === "const") {
            // make *const T parse the same as *T
            parserState.pos += 5;
            c = parserState.userQuery[parserState.pos];
        }
        while (c === " " && parserState.pos < parserState.length) {
            parserState.pos += 1;
            c = parserState.userQuery[parserState.pos];
        }
        if (!isEndCharacter(c) && parserState.pos < parserState.length) {
            getFilteredNextElem(query, parserState, generics, isInGenerics);
        }
        elems.push(makePrimitiveElement(name, { generics }));
    };

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
        handleRefOrPtr("&", "reference");
    } else if (parserState.userQuery[parserState.pos] === "*") {
        handleRefOrPtr("*", "pointer");
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
        this.elemCount = 0;
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
        this.elemCount += 1;
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

/** @type {Array<string>} */
const EMPTY_STRING_ARRAY = [];

/** @type {Array<rustdoc.FunctionType>} */
const EMPTY_GENERICS_ARRAY = [];

/** @type {Array<[number, rustdoc.FunctionType[]]>} */
const EMPTY_BINDINGS_ARRAY = [];

/** @type {Map<number, Array<any>>} */
const EMPTY_BINDINGS_MAP = new Map();

/**
 * @param {string|null} typename
 * @returns {number}
 */
function itemTypeFromName(typename) {
    if (typename === null) {
        return NO_TYPE_FILTER;
    }
    const index = itemTypes.findIndex(i => i === typename);
    if (index < 0) {
        throw ["Unknown type filter ", typename];
    }
    return index;
}

class DocSearch {
    /**
     * @param {string} rootPath
     * @param {stringdex.Database} database
     */
    constructor(rootPath, database) {
        this.rootPath = rootPath;
        this.database = database;

        this.typeNameIdOfOutput = -1;
        this.typeNameIdOfArray = -1;
        this.typeNameIdOfSlice = -1;
        this.typeNameIdOfArrayOrSlice = -1;
        this.typeNameIdOfTuple = -1;
        this.typeNameIdOfUnit = -1;
        this.typeNameIdOfTupleOrUnit = -1;
        this.typeNameIdOfReference = -1;
        this.typeNameIdOfPointer = -1;
        this.typeNameIdOfHof = -1;

        this.utf8decoder = new TextDecoder();

        /** @type {Map<number|null, rustdoc.FunctionType>} */
        this.TYPES_POOL = new Map();
    }

    /**
     * Load search index. If you do not call this function, `execQuery`
     * will never fulfill.
     */
    async buildIndex() {
        const nn = this.database.getData("normalizedName");
        if (!nn) {
            return;
        }
        // Each of these identifiers are used specially by
        // type-driven search.
        const [
            // output is the special associated type that goes
            // after the arrow: the type checker desugars
            // the path `Fn(a) -> b` into `Fn<Output=b, (a)>`
            output,
            // fn, fnmut, and fnonce all match `->`
            fn,
            fnMut,
            fnOnce,
            hof,
            // array and slice both match `[]`
            array,
            slice,
            arrayOrSlice,
            // tuple and unit both match `()`
            tuple,
            unit,
            tupleOrUnit,
            // reference matches `&`
            reference,
            pointer,
            // never matches `!`
            never,
        ] = await Promise.all([
            nn.search("output"),
            nn.search("fn"),
            nn.search("fnmut"),
            nn.search("fnonce"),
            nn.search("->"),
            nn.search("array"),
            nn.search("slice"),
            nn.search("[]"),
            nn.search("tuple"),
            nn.search("unit"),
            nn.search("()"),
            nn.search("reference"),
            nn.search("pointer"),
            nn.search("never"),
        ]);
        /**
         * @param {stringdex.Trie|null|undefined} trie
         * @param {rustdoc.ItemType} ty
         * @param {string} modulePath
         * @returns {Promise<number>}
         * */
        const first = async(trie, ty, modulePath) => {
            if (trie) {
                for (const id of trie.matches().entries()) {
                    const pathData = await this.getPathData(id);
                    if (pathData && pathData.ty === ty && pathData.modulePath === modulePath) {
                        return id;
                    }
                }
            }
            return -1;
        };
        this.typeNameIdOfOutput = await first(output, TY_ASSOCTYPE, "");
        this.typeNameIdOfFnPtr = await first(fn, TY_PRIMITIVE, "");
        this.typeNameIdOfFn = await first(fn, TY_TRAIT, "core::ops");
        this.typeNameIdOfFnMut = await first(fnMut, TY_TRAIT, "core::ops");
        this.typeNameIdOfFnOnce = await first(fnOnce, TY_TRAIT, "core::ops");
        this.typeNameIdOfArray = await first(array, TY_PRIMITIVE, "");
        this.typeNameIdOfSlice = await first(slice, TY_PRIMITIVE, "");
        this.typeNameIdOfArrayOrSlice = await first(arrayOrSlice, TY_PRIMITIVE, "");
        this.typeNameIdOfTuple = await first(tuple, TY_PRIMITIVE, "");
        this.typeNameIdOfUnit = await first(unit, TY_PRIMITIVE, "");
        this.typeNameIdOfTupleOrUnit = await first(tupleOrUnit, TY_PRIMITIVE, "");
        this.typeNameIdOfReference = await first(reference, TY_PRIMITIVE, "");
        this.typeNameIdOfPointer = await first(pointer, TY_PRIMITIVE, "");
        this.typeNameIdOfHof = await first(hof, TY_PRIMITIVE, "");
        this.typeNameIdOfNever = await first(never, TY_PRIMITIVE, "");
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
                    let extra = EMPTY_STRING_ARRAY;
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

            // Scan for invalid type filters, so that we can report the error
            // outside the search loop.
            /** @param {rustdoc.ParserQueryElement} elem */
            const checkTypeFilter = elem => {
                const ty = itemTypeFromName(elem.typeFilter);
                if (ty === TY_GENERIC && elem.generics.length !== 0) {
                    throw [
                        "Generic type parameter ",
                        elem.name,
                        " does not accept generic parameters",
                    ];
                }
                for (const generic of elem.generics) {
                    checkTypeFilter(generic);
                }
                for (const constraints of elem.bindings.values()) {
                    for (const constraint of constraints) {
                        checkTypeFilter(constraint);
                    }
                }
            };
            for (const elem of query.elems) {
                checkTypeFilter(elem);
            }
            for (const elem of query.returned) {
                checkTypeFilter(elem);
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
     * @param {number} id
     * @returns {Promise<string|null>}
     */
    async getName(id) {
        const ni = this.database.getData("name");
        if (!ni) {
            return null;
        }
        const name = await ni.at(id);
        return name === undefined || name === null ? null : this.utf8decoder.decode(name);
    }

    /**
     * @param {number} id
     * @returns {Promise<string|null>}
     */
    async getDesc(id) {
        const di = this.database.getData("desc");
        if (!di) {
            return null;
        }
        const desc = await di.at(id);
        return desc === undefined || desc === null ? null : this.utf8decoder.decode(desc);
    }

    /**
     * @param {number} id
     * @returns {Promise<number|null>}
     */
    async getAliasTarget(id) {
        const ai = this.database.getData("alias");
        if (!ai) {
            return null;
        }
        const bytes = await ai.at(id);
        if (bytes === undefined || bytes === null || bytes.length === 0) {
            return null;
        } else {
            /** @type {string} */
            const encoded = this.utf8decoder.decode(bytes);
            /** @type {number|null} */
            const decoded = JSON.parse(encoded);
            return decoded;
        }
    }

    /**
     * @param {number} id
     * @returns {Promise<rustdoc.EntryData|null>}
     */
    async getEntryData(id) {
        const ei = this.database.getData("entry");
        if (!ei) {
            return null;
        }
        const encoded = this.utf8decoder.decode(await ei.at(id));
        if (encoded === "" || encoded === undefined || encoded === null) {
            return null;
        }
        /**
         * krate,
         * ty,
         * module_path,
         * exact_module_path,
         * parent,
         * deprecated,
         * associated_item_disambiguator
         * @type {rustdoc.ArrayWithOptionals<[
         *     number,
         *     rustdoc.ItemType,
         *     number,
         *     number,
         *     number,
         *     number,
         * ], [string]>}
         */
        const raw = JSON.parse(encoded);
        return {
            krate: raw[0],
            ty: raw[1],
            modulePath: raw[2] === 0 ? null : raw[2] - 1,
            exactModulePath: raw[3] === 0 ? null : raw[3] - 1,
            parent: raw[4] === 0 ? null : raw[4] - 1,
            deprecated: raw[5] === 1 ? true : false,
            associatedItemDisambiguator: raw.length === 6 ? null : raw[6],
        };
    }

    /**
     * @param {number} id
     * @returns {Promise<rustdoc.PathData|null>}
     */
    async getPathData(id) {
        const pi = this.database.getData("path");
        if (!pi) {
            return null;
        }
        const encoded = this.utf8decoder.decode(await pi.at(id));
        if (encoded === "" || encoded === undefined || encoded === null) {
            return null;
        }
        /**
         * ty, module_path, exact_module_path, search_unbox, inverted_function_signature_index
         * @type {rustdoc.ArrayWithOptionals<[rustdoc.ItemType, string], [string|0, 0|1, string]>}
         */
        const raw = JSON.parse(encoded);
        return {
            ty: raw[0],
            modulePath: raw[1],
            exactModulePath: raw[2] === 0 || raw[2] === undefined ? raw[1] : raw[2],
        };
    }

    /**
     * @param {number} id
     * @returns {Promise<rustdoc.FunctionData|null>}
     */
    async getFunctionData(id) {
        const fi = this.database.getData("function");
        if (!fi) {
            return null;
        }
        const encoded = this.utf8decoder.decode(await fi.at(id));
        if (encoded === "" || encoded === undefined || encoded === null) {
            return null;
        }
        /**
         * function_signature, param_names
         * @type {[string, string[]]}
         */
        const raw = JSON.parse(encoded);

        const parser = new VlqHexDecoder(raw[0], async functionSearchType => {
            if (typeof functionSearchType === "number") {
                return null;
            }
            const INPUTS_DATA = 0;
            const OUTPUT_DATA = 1;
            /** @type {Promise<rustdoc.FunctionType[]>} */
            let inputs_;
            /** @type {Promise<rustdoc.FunctionType[]>} */
            let output_;
            if (typeof functionSearchType[INPUTS_DATA] === "number") {
                inputs_ = Promise.all([
                    this.buildItemSearchType(functionSearchType[INPUTS_DATA]),
                ]);
            } else {
                // @ts-ignore
                inputs_ = this.buildItemSearchTypeAll(functionSearchType[INPUTS_DATA]);
            }
            if (functionSearchType.length > 1) {
                if (typeof functionSearchType[OUTPUT_DATA] === "number") {
                    output_ = Promise.all([
                        this.buildItemSearchType(functionSearchType[OUTPUT_DATA]),
                    ]);
                } else {
                    // @ts-expect-error
                    output_ = this.buildItemSearchTypeAll(functionSearchType[OUTPUT_DATA]);
                }
            } else {
                output_ = Promise.resolve(EMPTY_GENERICS_ARRAY);
            }
            /** @type {Promise<rustdoc.FunctionType[]>[]} */
            const where_clause_ = [];
            const l = functionSearchType.length;
            for (let i = 2; i < l; ++i) {
                where_clause_.push(typeof functionSearchType[i] === "number"
                    // @ts-expect-error
                    ? Promise.all([this.buildItemSearchType(functionSearchType[i])])
                    // @ts-expect-error
                    : this.buildItemSearchTypeAll(functionSearchType[i]),
                );
            }
            const [inputs, output, where_clause] = await Promise.all([
                inputs_,
                output_,
                Promise.all(where_clause_),
            ]);
            return {
                inputs, output, where_clause,
            };
        });

        return {
            functionSignature: await parser.next(),
            paramNames: raw[1],
            elemCount: parser.elemCount,
        };
    }

    /**
     * @param {number} id
     * @returns {Promise<rustdoc.TypeData|null>}
     */
    async getTypeData(id) {
        const ti = this.database.getData("type");
        if (!ti) {
            return null;
        }
        const encoded = this.utf8decoder.decode(await ti.at(id));
        if (encoded === "" || encoded === undefined || encoded === null) {
            return null;
        }
        /**
         * function_signature, param_names
         * @type {[string, string, number] | [string, string] | [] | null}
         */
        const raw = JSON.parse(encoded);

        if (!raw || raw.length === 0) {
            return null;
        }

        let searchUnbox = false;
        const invertedFunctionInputsIndex = [];
        const invertedFunctionOutputIndex = [];

        if (typeof raw[0] === "string") {
            if (raw[2]) {
                searchUnbox = true;
            }
            // the inverted function signature index is a list of bitmaps,
            // by number of types that appear in the function
            let i = 0;
            let pb = makeUint8ArrayFromBase64(raw[0]);
            let l = pb.length;
            while (i < l) {
                if (pb[i] === 0) {
                    invertedFunctionInputsIndex.push(RoaringBitmap.empty());
                    i += 1;
                } else {
                    const bitmap = new RoaringBitmap(pb, i);
                    i += bitmap.consumed_len_bytes;
                    invertedFunctionInputsIndex.push(bitmap);
                }
            }
            i = 0;
            pb = makeUint8ArrayFromBase64(raw[1]);
            l = pb.length;
            while (i < l) {
                if (pb[i] === 0) {
                    invertedFunctionOutputIndex.push(RoaringBitmap.empty());
                    i += 1;
                } else {
                    const bitmap = new RoaringBitmap(pb, i);
                    i += bitmap.consumed_len_bytes;
                    invertedFunctionOutputIndex.push(bitmap);
                }
            }
        } else if (raw[0]) {
            searchUnbox = true;
        }

        return { searchUnbox, invertedFunctionInputsIndex, invertedFunctionOutputIndex };
    }

    /**
     * @returns {Promise<string[]>}
     */
    async getCrateNameList() {
        const crateNames = this.database.getData("crateNames");
        if (!crateNames) {
            return [];
        }
        const l = crateNames.length;
        const names = [];
        for (let i = 0; i < l; ++i) {
            names.push(crateNames.at(i).then(name => {
                if (name === undefined) {
                    return "";
                }
                return this.utf8decoder.decode(name);
            }));
        }
        return Promise.all(names);
    }

    /**
     * @param {number} id non-negative generic index
     * @returns {Promise<stringdex.RoaringBitmap[]>}
     */
    async getGenericInvertedIndex(id) {
        const gii = this.database.getData("generic_inverted_index");
        if (!gii) {
            return [];
        }
        const pb = await gii.at(id);
        if (pb === undefined || pb === null || pb.length === 0) {
            return [];
        }

        const invertedFunctionSignatureIndex = [];
        // the inverted function signature index is a list of bitmaps,
        // by number of types that appear in the function
        let i = 0;
        const l = pb.length;
        while (i < l) {
            if (pb[i] === 0) {
                invertedFunctionSignatureIndex.push(RoaringBitmap.empty());
                i += 1;
            } else {
                const bitmap = new RoaringBitmap(pb, i);
                i += bitmap.consumed_len_bytes;
                invertedFunctionSignatureIndex.push(bitmap);
            }
        }
        return invertedFunctionSignatureIndex;
    }

    /**
     * @param {number} id
     * @param {boolean} loadFunctionData
     * @returns {Promise<rustdoc.Row?>}
     */
    async getRow(id, loadFunctionData) {
        const [name_, entry, path, functionData] = await Promise.all([
            this.getName(id),
            this.getEntryData(id),
            this.getPathData(id),
            loadFunctionData ? this.getFunctionData(id) : null,
        ]);
        if (!entry && !path) {
            return null;
        }
        const [
            moduleName,
            modulePathData,
            exactModuleName,
            exactModulePathData,
        ] = await Promise.all([
            entry && entry.modulePath !== null ? this.getName(entry.modulePath) : null,
            entry && entry.modulePath !== null ? this.getPathData(entry.modulePath) : null,
            entry && entry.exactModulePath !== null ?
                this.getName(entry.exactModulePath) :
                null,
            entry && entry.exactModulePath !== null ?
                this.getPathData(entry.exactModulePath) :
                null,
        ]);
        const name = name_ === null ? "" : name_;
        const normalizedName = (name.indexOf("_") === -1 ?
            name :
            name.replace(/_/g, "")).toLowerCase();
        const modulePath = modulePathData === null || moduleName === null ? "" :
            (modulePathData.modulePath === "" ?
                moduleName :
                `${modulePathData.modulePath}::${moduleName}`);
        const [parentName, parentPath] = entry !== null && entry.parent !== null ?
            await Promise.all([this.getName(entry.parent), this.getPathData(entry.parent)]) :
            [null, null];
        return {
            id,
            crate: entry ? nonnull(await this.getName(entry.krate)) : "",
            ty: entry ? entry.ty : nonnull(path).ty,
            name,
            normalizedName,
            modulePath,
            exactModulePath: exactModulePathData === null || exactModuleName === null ? modulePath :
                (exactModulePathData.exactModulePath === "" ?
                    exactModuleName :
                    `${exactModulePathData.exactModulePath}::${exactModuleName}`),
            entry,
            path,
            functionData,
            deprecated: entry ? entry.deprecated : false,
            parent: parentName !== null && parentPath !== null ?
                { name: parentName, path: parentPath } :
                null,
        };
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
     *
     * @return {Promise<Array<rustdoc.FunctionType>>}
     */
    async buildItemSearchTypeAll(types) {
        return types && types.length > 0 ?
            await Promise.all(types.map(type => this.buildItemSearchType(type))) :
            EMPTY_GENERICS_ARRAY;
    }

    /**
     * Converts a single type.
     *
     * @param {rustdoc.RawFunctionType} type
     * @return {Promise<rustdoc.FunctionType>}
     */
    async buildItemSearchType(type) {
        const PATH_INDEX_DATA = 0;
        const GENERICS_DATA = 1;
        const BINDINGS_DATA = 2;
        let id, generics;
        /**
         * @type {Map<number, rustdoc.FunctionType[]>}
         */
        let bindings;
        if (typeof type === "number") {
            id = type;
            generics = EMPTY_GENERICS_ARRAY;
            bindings = EMPTY_BINDINGS_MAP;
        } else {
            id = type[PATH_INDEX_DATA];
            generics = await this.buildItemSearchTypeAll(type[GENERICS_DATA]);
            if (type[BINDINGS_DATA] && type[BINDINGS_DATA].length > 0) {
                bindings = new Map((await Promise.all(type[BINDINGS_DATA].map(
                    /**
                     * @param {[rustdoc.RawFunctionType, rustdoc.RawFunctionType[]]} binding
                     * @returns {Promise<[number, rustdoc.FunctionType[]][]>}
                    */
                    async binding => {
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
                        const [k, v] = await Promise.all([
                            this.buildItemSearchType(assocType).then(t => t.id),
                            this.buildItemSearchTypeAll(constraints),
                        ]);
                        return k === null ? EMPTY_BINDINGS_ARRAY : [[k, v]];
                    },
                ))).flat());
            } else {
                bindings = EMPTY_BINDINGS_MAP;
            }
        }
        /**
         * @type {rustdoc.FunctionType}
         */
        let result;
        if (id < 0) {
            // types less than 0 are generic parameters
            // the actual names of generic parameters aren't stored, since they aren't API
            result = {
                id,
                name: "",
                ty: TY_GENERIC,
                path: null,
                exactPath: null,
                generics,
                bindings,
                unboxFlag: true,
            };
        } else if (id === 0) {
            // `0` is used as a sentinel because it's fewer bytes than `null`
            result = {
                id: null,
                name: "",
                ty: TY_GENERIC,
                path: null,
                exactPath: null,
                generics,
                bindings,
                unboxFlag: true,
            };
        } else {
            const [name, path, type] = await Promise.all([
                this.getName(id - 1),
                this.getPathData(id - 1),
                this.getTypeData(id - 1),
            ]);
            if (path === undefined || path === null || type === undefined || type === null) {
                return {
                    id: null,
                    name: "",
                    ty: TY_GENERIC,
                    path: null,
                    exactPath: null,
                    generics,
                    bindings,
                    unboxFlag: true,
                };
            }
            result = {
                id: id - 1,
                name,
                ty: path.ty,
                path: path.modulePath,
                exactPath: path.exactModulePath === null ? path.modulePath : path.exactModulePath,
                generics,
                bindings,
                unboxFlag: type.searchUnbox,
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
                    const v2 = result.bindings.get(k);
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
     * Executes the parsed query and builds a {ResultsTable}.
     *
     * @param  {rustdoc.ParsedQuery<rustdoc.ParserQueryElement>} parsedQuery
     *     - The parsed user query
     * @param  {Object} filterCrates - Crate to search in if defined
     * @param  {string} currentCrate - Current crate, to rank results from this crate higher
     *
     * @return {Promise<rustdoc.ResultsTable>}
     */
    async execQuery(parsedQuery, filterCrates, currentCrate) {
        const queryLen =
            parsedQuery.elems.reduce((acc, next) => acc + next.pathLast.length, 0) +
            parsedQuery.returned.reduce((acc, next) => acc + next.pathLast.length, 0);
        const maxEditDistance = Math.floor(queryLen / 3);

        /**
         * @param {rustdoc.Row} item
         * @returns {[string, string, string]}
         */
        const buildHrefAndPath = item => {
            let displayPath;
            let href;
            const type = itemTypes[item.ty];
            const name = item.name;
            let path = item.modulePath;
            let exactPath = item.exactModulePath;

            if (type === "mod") {
                displayPath = path + "::";
                href = this.rootPath + path.replace(/::/g, "/") + "/" +
                    name + "/index.html";
            } else if (type === "import") {
                displayPath = item.modulePath + "::";
                href = this.rootPath + item.modulePath.replace(/::/g, "/") +
                    "/index.html#reexport." + name;
            } else if (type === "primitive" || type === "keyword" || type === "attribute") {
                displayPath = "";
                exactPath = "";
                href = this.rootPath + path.replace(/::/g, "/") +
                    "/" + type + "." + name + ".html";
            } else if (type === "externcrate") {
                displayPath = "";
                href = this.rootPath + name + "/index.html";
            } else if (item.parent) {
                const myparent = item.parent;
                let anchor = type + "." + name;
                const parentType = itemTypes[myparent.path.ty];
                let pageType = parentType;
                let pageName = myparent.name;
                exactPath = `${myparent.path.exactModulePath}::${myparent.name}`;

                if (parentType === "primitive") {
                    displayPath = myparent.name + "::";
                    exactPath = myparent.name;
                } else if (type === "structfield" && parentType === "variant") {
                    // Structfields belonging to variants are special: the
                    // final path element is the enum name.
                    const enumNameIdx = item.modulePath.lastIndexOf("::");
                    const enumName = item.modulePath.substr(enumNameIdx + 2);
                    path = item.modulePath.substr(0, enumNameIdx);
                    displayPath = path + "::" + enumName + "::" + myparent.name + "::";
                    anchor = "variant." + myparent.name + ".field." + name;
                    pageType = "enum";
                    pageName = enumName;
                } else {
                    displayPath = path + "::" + myparent.name + "::";
                }
                if (item.entry && item.entry.associatedItemDisambiguator !== null) {
                    anchor = item.entry.associatedItemDisambiguator + "/" + anchor;
                }
                href = this.rootPath + path.replace(/::/g, "/") +
                    "/" + pageType +
                    "." + pageName +
                    ".html#" + anchor;
            } else {
                displayPath = item.modulePath + "::";
                href = this.rootPath + item.modulePath.replace(/::/g, "/") +
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
         * The output is formatted as an array of hunks, where odd numbered
         * hunks are highlighted and even numbered ones are not.
         *
         * @param {rustdoc.ResultObject} obj
         * @param {"sig"|"elems"|"returned"|null} typeInfo
         * @param {rustdoc.QueryElement[]} elems
         * @param {rustdoc.QueryElement[]} returned
         * @returns {Promise<rustdoc.DisplayTypeSignature>}
         */
        const formatDisplayTypeSignature = async(obj, typeInfo, elems, returned) => {
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
                    elems,
                    objType.where_clause,
                    null,
                    mgensScratch => {
                        fnOutput = unifyFunctionTypes(
                            objType.output,
                            returned,
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
                const highlighted = unifyFunctionTypes(
                    typeInfo === "elems" ? objType.inputs : objType.output,
                    typeInfo === "elems" ? elems : returned,
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

            elems.forEach(remapQuery);
            returned.forEach(remapQuery);

            /**
             * Write text to a highlighting array.
             * Index 0 is not highlighted, index 1 is highlighted,
             * index 2 is not highlighted, etc.
             *
             * @param {{name: string|null, highlighted?: boolean}} fnType - input
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
             * @returns {Promise<void>}
             */
            const writeHof = async(fnType, result) => {
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
                    await writeFn(fnType, result);
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
                    await writeFn(fnType, result);
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
             * @returns {Promise<boolean>}
             */
            const writeSpecialPrimitive = async(fnType, result) => {
                if (fnType.id === this.typeNameIdOfArray || fnType.id === this.typeNameIdOfSlice ||
                    fnType.id === this.typeNameIdOfTuple || fnType.id === this.typeNameIdOfUnit) {
                    const [ob, sb] =
                        fnType.id === this.typeNameIdOfArray ||
                            fnType.id === this.typeNameIdOfSlice ?
                        ["[", "]"] :
                        ["(", ")"];
                    pushText({ name: ob, highlighted: fnType.highlighted }, result);
                    await onEachBtwnAsync(
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
                    await onEachBtwnAsync(
                        fnType.generics,
                        async value => {
                            prevHighlighted = !!value.highlighted;
                            await writeFn(value, result);
                        },
                        // @ts-expect-error
                        value => pushText({
                            name: " ",
                            highlighted: prevHighlighted && value.highlighted,
                        }, result),
                    );
                    return true;
                } else if (fnType.id === this.typeNameIdOfPointer) {
                    pushText({ name: "*", highlighted: fnType.highlighted }, result);
                    if (fnType.generics.length < 2) {
                        pushText({ name: "const ", highlighted: fnType.highlighted }, result);
                    }
                    let prevHighlighted = false;
                    await onEachBtwnAsync(
                        fnType.generics,
                        async value => {
                            prevHighlighted = !!value.highlighted;
                            await writeFn(value, result);
                        },
                        // @ts-expect-error
                        value => pushText({
                            name: " ",
                            highlighted: prevHighlighted && value.highlighted,
                        }, result),
                    );
                    return true;
                } else if (
                    fnType.id === this.typeNameIdOfFn ||
                    fnType.id === this.typeNameIdOfFnMut ||
                    fnType.id === this.typeNameIdOfFnOnce ||
                    fnType.id === this.typeNameIdOfFnPtr
                ) {
                    await writeHof(fnType, result);
                    return true;
                } else if (fnType.id === this.typeNameIdOfNever) {
                    pushText({ name: "!", highlighted: fnType.highlighted }, result);
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
             * @returns {Promise<void>}
             */
            const writeFn = async(fnType, result) => {
                if (fnType.id !== null && fnType.id < 0) {
                    if (fnParamNames[-1 - fnType.id] === "") {
                        // Normally, there's no need to shown an unhighlighted
                        // where clause, but if it's impl Trait, then we do.
                        const generics = fnType.generics.length > 0 ?
                            fnType.generics :
                            objType.where_clause[-1 - fnType.id];
                        for (const nested of generics) {
                            await writeFn(nested, result);
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
                    await onEachBtwnAsync(
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
                        if (await writeSpecialPrimitive(fnType, result)) {
                            return;
                        }
                    } else if (fnType.ty === TY_TRAIT && (
                        fnType.id === this.typeNameIdOfFn ||
                        fnType.id === this.typeNameIdOfFnMut ||
                        fnType.id === this.typeNameIdOfFnOnce ||
                        fnType.id === this.typeNameIdOfFnPtr
                    )) {
                        await writeHof(fnType, result);
                        return;
                    } else if (fnType.name === "" &&
                        fnType.bindings.size === 0 &&
                        fnType.generics.length !== 0
                    ) {
                        pushText({ name: "impl ", highlighted: false }, result);
                        if (fnType.generics.length > 1) {
                            pushText({ name: "(", highlighted: false }, result);
                        }
                        await onEachBtwnAsync(
                            fnType.generics,
                            value => writeFn(value, result),
                            // @ts-expect-error
                            () => pushText({ name: ", ",  highlighted: false }, result),
                        );
                        if (fnType.generics.length > 1) {
                            pushText({ name: ")", highlighted: false }, result);
                        }
                        return;
                    }
                    pushText(fnType, result);
                    let hasBindings = false;
                    if (fnType.bindings.size > 0) {
                        await onEachBtwnAsync(
                            await Promise.all([...fnType.bindings.entries()].map(
                                /**
                                 * @param {[number, rustdoc.HighlightedFunctionType[]]} param0
                                 * @returns {Promise<[
                                 *     string|null,
                                 *     rustdoc.HighlightedFunctionType[],
                                 * ]>}
                                 */
                                async([key, values]) => [await this.getName(key), values],
                            )),
                            async([name, values]) => {
                                // @ts-expect-error
                                if (values.length === 1 && values[0].id < 0 &&
                                    // @ts-expect-error
                                    `${fnType.name}::${name}` === fnParamNames[-1 - values[0].id]
                                ) {
                                    // the internal `Item=Iterator::Item` type variable should be
                                    // shown in the where clause and name mapping output, but is
                                    // redundant in this spot
                                    for (const value of values) {
                                        await writeFn(value, []);
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
                                await onEachBtwnAsync(
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
                    await onEachBtwnAsync(
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
            await onEachBtwnAsync(
                fnInputs,
                fnType => writeFn(fnType, type),
                // @ts-expect-error
                () => pushText({ name: ", ",  highlighted: false }, type),
            );
            pushText({ name: " -> ", highlighted: false }, type);
            await onEachBtwnAsync(
                fnOutput,
                fnType => writeFn(fnType, type),
                // @ts-expect-error
                () => pushText({ name: ", ",  highlighted: false }, type),
            );

            return {type, mappedNames, whereClause};
        };

        /**
         * Add extra data to result objects, and filter items that have been
         * marked for removal.
         *
         * @param {[rustdoc.PlainResultObject, rustdoc.Row][]} results
         * @param {"sig"|"elems"|"returned"|null} typeInfo
         * @param {Set<string>} duplicates
         * @returns {rustdoc.ResultObject[]}
         */
        const transformResults = (results, typeInfo, duplicates) => {
            const out = [];

            for (const [result, item] of results) {
                if (item.id !== -1) {
                    const res = buildHrefAndPath(item);
                    // many of these properties don't strictly need to be
                    // copied over, but copying them over satisfies tsc,
                    // and hopefully plays nice with the shape optimization
                    // of the browser engine.
                    /** @type {rustdoc.ResultObject} */
                    const obj = Object.assign({
                        parent: item.parent ? {
                            path: item.parent.path.modulePath,
                            exactPath: item.parent.path.exactModulePath ||
                                item.parent.path.modulePath,
                            name: item.parent.name,
                            ty: item.parent.path.ty,
                        } : undefined,
                        type: item.functionData && item.functionData.functionSignature ?
                            item.functionData.functionSignature :
                            undefined,
                        paramNames: item.functionData && item.functionData.paramNames ?
                            item.functionData.paramNames :
                            undefined,
                        dist: result.dist,
                        path_dist: result.path_dist,
                        index: result.index,
                        desc: this.getDesc(result.id),
                        item,
                        displayPath: pathSplitter(res[0]),
                        fullPath: "",
                        href: "",
                        displayTypeSignature: null,
                    }, result);

                    // To be sure than it some items aren't considered as duplicate.
                    obj.fullPath = res[2] + "|" + obj.item.ty;

                    if (duplicates.has(obj.fullPath)) {
                        continue;
                    }

                    // Exports are specifically not shown if the items they point at
                    // are already in the results.
                    if (obj.item.ty === TY_IMPORT && duplicates.has(res[2])) {
                        continue;
                    }
                    if (duplicates.has(res[2] + "|" + TY_IMPORT)) {
                        continue;
                    }
                    duplicates.add(obj.fullPath);
                    duplicates.add(res[2]);

                    if (typeInfo !== null) {
                        obj.displayTypeSignature = formatDisplayTypeSignature(
                            obj,
                            typeInfo,
                            result.elems,
                            result.returned,
                        );
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

        const sortAndTransformResults =
            /**
             * @this {DocSearch}
             * @param {Array<rustdoc.PlainResultObject|null>} results
             * @param {"sig"|"elems"|"returned"|null} typeInfo
             * @param {string} preferredCrate
             * @param {Set<string>} duplicates
             * @returns {AsyncGenerator<rustdoc.ResultObject, number>}
             */
            async function*(results, typeInfo, preferredCrate, duplicates) {
                const userQuery = parsedQuery.userQuery;
                const normalizedUserQuery = parsedQuery.userQuery.toLowerCase();
                const isMixedCase = normalizedUserQuery !== userQuery;
                /**
                 * @type {[rustdoc.PlainResultObject, rustdoc.Row][]}
                 */
                const result_list = [];
                for (const result of results.values()) {
                    if (!result) {
                        continue;
                    }
                    /**
                     * @type {rustdoc.Row?}
                     */
                    const item = await this.getRow(result.id, typeInfo !== null);
                    if (!item) {
                        continue;
                    }
                    if (filterCrates !== null && item.crate !== filterCrates) {
                        continue;
                    }
                    if (item) {
                        result_list.push([result, item]);
                    } else {
                        continue;
                    }
                }

                result_list.sort(([aaa, aai], [bbb, bbi]) => {
                    /** @type {number} */
                    let a;
                    /** @type {number} */
                    let b;

                    if (typeInfo === null) {
                        // in name based search...

                        // sort by exact case-sensitive match
                        if (isMixedCase) {
                            a = Number(aai.name !== userQuery);
                            b = Number(bbi.name !== userQuery);
                            if (a !== b) {
                                return a - b;
                            }
                        }

                        // sort by exact match with regard to the last word (mismatch goes later)
                        a = Number(aai.normalizedName !== normalizedUserQuery);
                        b = Number(bbi.normalizedName !== normalizedUserQuery);
                        if (a !== b) {
                            return a - b;
                        }

                        // sort by index of keyword in item name (no literal occurrence goes later)
                        a = Number(aaa.index < 0);
                        b = Number(bbb.index < 0);
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

                    // sort aliases lower
                    a = Number(aaa.is_alias);
                    b = Number(bbb.is_alias);
                    if (a !== b) {
                        return a - b;
                    }

                    // sort deprecated items later
                    a = Number(aai.deprecated);
                    b = Number(bbi.deprecated);
                    if (a !== b) {
                        return a - b;
                    }

                    // sort by crate (current crate comes first)
                    a = Number(aai.crate !== preferredCrate);
                    b = Number(bbi.crate !== preferredCrate);
                    if (a !== b) {
                        return a - b;
                    }

                    // sort by item name length (longer goes later)
                    a = Number(aai.normalizedName.length);
                    b = Number(bbi.normalizedName.length);
                    if (a !== b) {
                        return a - b;
                    }

                    // sort by item name (lexicographically larger goes later)
                    let aw = aai.normalizedName;
                    let bw = bbi.normalizedName;
                    if (aw !== bw) {
                        return (aw > bw ? +1 : -1);
                    }

                    // sort by description (no description goes later)
                    const di = this.database.getData("desc");
                    if (di) {
                        a = Number(di.isEmpty(aaa.id));
                        b = Number(di.isEmpty(bbb.id));
                        if (a !== b) {
                            return a - b;
                        }
                    }

                    // sort by type (later occurrence in `itemTypes` goes later)
                    a = Number(aai.ty);
                    b = Number(bbi.ty);
                    if (a !== b) {
                        return a - b;
                    }

                    // sort by path (lexicographically larger goes later)
                    const ap = aai.modulePath;
                    const bp = bbi.modulePath;
                    aw = ap === undefined ? "" : ap;
                    bw = bp === undefined ? "" : bp;
                    if (aw !== bw) {
                        return (aw > bw ? +1 : -1);
                    }

                    // que sera, sera
                    return 0;
                });

                const transformed_result_list = transformResults(result_list, typeInfo, duplicates);
                yield* transformed_result_list;
                return transformed_result_list.length;
            }
            .bind(this);

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
                // For these special cases, matching code need added to the inverted index.
                // search_index.rs -> convert_render_type does this
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
                } else if (queryElem.id === this.typeNameIdOfHof && (
                    fnType.id === this.typeNameIdOfFn ||
                    fnType.id === this.typeNameIdOfFnMut ||
                    fnType.id === this.typeNameIdOfFnOnce ||
                    fnType.id === this.typeNameIdOfFnPtr
                )) {
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
         * @param {rustdoc.QueryElement[]} elems
         * @param {rustdoc.FunctionType[]} list    - A list of function types.
         * @param {rustdoc.FunctionType[][]} where_clause - Trait bounds for generic items.
         */
        function containsTypeFromQuery(elems, list, where_clause) {
            if (!list) return false;
            for (const ty of elems) {
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
         * @param {string[]} path indexed page path
         * @returns {null|number} edit distance
         */
        function checkPath(contains, path) {
            if (contains.length === 0) {
                return 0;
            }
            const maxPathEditDistance = parsedQuery.literalSearch ? 0 : Math.floor(
                contains.reduce((acc, next) => acc + next.length, 0) / 3,
            );
            let ret_dist = maxPathEditDistance + 1;

            const length = path.length;
            const clength = contains.length;
            pathiter: for (let i = length - clength; i >= 0; i -= 1) {
                let dist_total = 0;
                for (let x = 0; x < clength; ++x) {
                    const [p, c] = [path[i + x], contains[x]];
                    if (parsedQuery.literalSearch && p !== c) {
                        continue pathiter;
                    } else if (Math.floor((p.length - c.length) / 3) <= maxPathEditDistance &&
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

        /**
         * Compute an "edit distance" that ignores missing path elements.
         * @param {string[]} contains search query path
         * @param {rustdoc.Row} row indexed item
         * @returns {null|number} edit distance
         */
        function checkRowPath(contains, row) {
            if (contains.length === 0) {
                return 0;
            }

            const path = row.modulePath.split("::");

            if (row.parent && row.parent.name) {
                path.push(row.parent.name.toLowerCase());
            }

            return checkPath(contains, path);
        }

        /**
         *
         * @param {number} filter
         * @param {rustdoc.ItemType} type
         * @returns
         */
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

        const innerRunNameQuery =
            /**
             * @this {DocSearch}
             * @param {string} currentCrate
             * @returns {AsyncGenerator<rustdoc.ResultObject>}
             */
            async function*(currentCrate) {
                const index = this.database.getData("normalizedName");
                if (!index) {
                    return;
                }
                const idDuplicates = new Set();
                const pathDuplicates = new Set();
                let count = 0;
                const prefixResults = [];
                const normalizedUserQuery = parsedQuery.userQuery
                    .replace(/[_"]/g, "")
                    .toLowerCase();
                /**
                 * @param {string} name
                 * @param {number} alias
                 * @param {number} dist
                 * @param {number} index
                 * @returns {Promise<rustdoc.PlainResultObject?>}
                 */
                const handleAlias = async(name, alias, dist, index) => {
                    return {
                        id: alias,
                        dist,
                        path_dist: 0,
                        index,
                        alias: name,
                        is_alias: true,
                        elems: [], // only used in type-based queries
                        returned: [], // only used in type-based queries
                        original: await this.getRow(alias, false),
                    };
                };
                /**
                 * @param {Promise<rustdoc.PlainResultObject|null>[]} data
                 * @returns {AsyncGenerator<rustdoc.ResultObject, boolean>}
                 */
                const flush = async function* (data) {
                    const satr = sortAndTransformResults(
                        await Promise.all(data),
                        null,
                        currentCrate,
                        pathDuplicates,
                    );
                    data.length = 0;
                    for await (const processed of satr) {
                        yield processed;
                        count += 1;
                        if ((count & 0x7F) === 0) {
                            await yieldToBrowser();
                        }
                        if (count >= MAX_RESULTS) {
                            return true;
                        }
                    }
                    return false;
                };
                const aliasResults = await index.search(normalizedUserQuery);
                if (aliasResults) {
                    for (const id of aliasResults.matches().entries()) {
                        const [name, alias] = await Promise.all([
                            this.getName(id),
                            this.getAliasTarget(id),
                        ]);
                        if (name !== null &&
                            alias !== null &&
                            !idDuplicates.has(id) &&
                            name.replace(/[_"]/g, "").toLowerCase() === normalizedUserQuery
                        ) {
                            prefixResults.push(handleAlias(name, alias, 0, 0));
                            idDuplicates.add(id);
                        }
                    }
                }
                if (parsedQuery.error !== null || parsedQuery.elems.length === 0) {
                    yield* flush(prefixResults);
                    return;
                }
                const elem = parsedQuery.elems[0];
                const typeFilter = itemTypeFromName(elem.typeFilter);
                /**
                 * @param {number} id
                 * @returns {Promise<rustdoc.PlainResultObject?>}
                 */
                const handleNameSearch = async id => {
                    const row = await this.getRow(id, false);
                    if (!row || !row.entry) {
                        return null;
                    }
                    if (!typePassesFilter(typeFilter, row.ty) ||
                        (filterCrates !== null && row.crate !== filterCrates)) {
                        return null;
                    }

                    /** @type {number|null} */
                    let pathDist = 0;
                    if (elem.fullPath.length > 1) {
                        pathDist = checkRowPath(elem.pathWithoutLast, row);
                        if (pathDist === null) {
                            return null;
                        }
                    }

                    if (parsedQuery.literalSearch) {
                        return row.name.toLowerCase() === elem.pathLast ? {
                            id,
                            dist: 0,
                            path_dist: 0,
                            index: 0,
                            elems: [], // only used in type-based queries
                            returned: [], // only used in type-based queries
                            is_alias: false,
                        } : null;
                    } else {
                        return {
                            id,
                            dist: editDistance(
                                row.normalizedName,
                                elem.normalizedPathLast,
                                maxEditDistance,
                            ),
                            path_dist: pathDist,
                            index: row.normalizedName.indexOf(elem.normalizedPathLast),
                            elems: [], // only used in type-based queries
                            returned: [], // only used in type-based queries
                            is_alias: false,
                        };
                    }
                };
                if (elem.normalizedPathLast === "") {
                    // faster full-table scan for this specific case.
                    const l = index.length;
                    for (let id = 0; id < l; ++id) {
                        if (!idDuplicates.has(id)) {
                            idDuplicates.add(id);
                            prefixResults.push(handleNameSearch(id));
                        }
                        if (yield* flush(prefixResults)) {
                            return;
                        }
                    }
                    return;
                }
                const results = await index.search(elem.normalizedPathLast);
                if (results) {
                    for await (const result of results.prefixMatches()) {
                        for (const id of result.entries()) {
                            if (!idDuplicates.has(id)) {
                                idDuplicates.add(id);
                                prefixResults.push(handleNameSearch(id));
                                const [name, alias] = await Promise.all([
                                    this.getName(id),
                                    this.getAliasTarget(id),
                                ]);
                                if (name !== null && alias !== null) {
                                    prefixResults.push(handleAlias(name, alias, 0, 0));
                                }
                            }
                        }
                        if (yield* flush(prefixResults)) {
                            return;
                        }
                    }
                    if (yield* flush(prefixResults)) {
                        return;
                    }
                }
                const levSearchResults = index.searchLev(elem.normalizedPathLast);
                const levResults = [];
                for await (const levResult of levSearchResults) {
                    for (const id of levResult.matches().entries()) {
                        if (!idDuplicates.has(id)) {
                            idDuplicates.add(id);
                            levResults.push(handleNameSearch(id));
                            const [name, alias] = await Promise.all([
                                this.getName(id),
                                this.getAliasTarget(id),
                            ]);
                            if (name !== null && alias !== null) {
                                levResults.push(handleAlias(
                                    name,
                                    alias,
                                    editDistance(elem.normalizedPathLast, name, maxEditDistance),
                                    name.indexOf(elem.normalizedPathLast),
                                ));
                            }
                        }
                    }
                }
                yield* flush(levResults);
                if (results) {
                    const substringResults = [];
                    for await (const result of results.substringMatches()) {
                        for (const id of result.entries()) {
                            if (!idDuplicates.has(id)) {
                                idDuplicates.add(id);
                                substringResults.push(handleNameSearch(id));
                                const [name, alias] = await Promise.all([
                                    this.getName(id),
                                    this.getAliasTarget(id),
                                ]);
                                if (name !== null && alias !== null) {
                                    levResults.push(handleAlias(
                                        name,
                                        alias,
                                        editDistance(
                                            elem.normalizedPathLast,
                                            name,
                                            maxEditDistance,
                                        ),
                                        name.indexOf(elem.normalizedPathLast),
                                    ));
                                }
                            }
                        }
                        if (yield* flush(substringResults)) {
                            return;
                        }
                    }
                }
            }
            .bind(this);

        const innerRunTypeQuery =
            /**
             * @this {DocSearch}
             * @param {rustdoc.ParserQueryElement[]} inputs
             * @param {rustdoc.ParserQueryElement[]} output
             * @param {"sig"|"elems"|"returned"|null} typeInfo
             * @param {string} currentCrate
             * @returns {AsyncGenerator<rustdoc.ResultObject>}
             */
            async function*(inputs, output, typeInfo, currentCrate) {
                const index = this.database.getData("normalizedName");
                if (!index) {
                    return;
                }
                /** @type {Map<string, number>} */
                const genericMap = new Map();
                /**
                 * @template Q
                 * @typedef {{
                 *     invertedIndex: stringdex.RoaringBitmap[],
                 *     queryElem: Q,
                 * }} PostingsList
                 */
                /** @type {stringdex.RoaringBitmap[]} */
                const empty_inverted_index = [];
                /** @type {PostingsList<any>[]} */
                const empty_postings_list = [];
                /** @type {stringdex.RoaringBitmap[]} */
                const everything_inverted_index = [];
                for (let i = 0; i < 64; ++i) {
                    everything_inverted_index.push(RoaringBitmap.everything());
                }
                /**
                 * @type {PostingsList<rustdoc.QueryElement[]>}
                 */
                const everything_postings_list = {
                    invertedIndex: everything_inverted_index,
                    queryElem: [],
                };
                /**
                 * @type {PostingsList<rustdoc.QueryElement[]>[]}
                 */
                const nested_everything_postings_list = [everything_postings_list];
                /**
                 * @param {...stringdex.RoaringBitmap[]} idx
                 * @returns {stringdex.RoaringBitmap[]}
                 */
                const intersectInvertedIndexes = (...idx) => {
                    let i = 0;
                    const l = idx.length;
                    while (i < l - 1 && idx[i] === everything_inverted_index) {
                        i += 1;
                    }
                    const result = [...idx[i]];
                    for (; i < l; ++i) {
                        if (idx[i] === everything_inverted_index) {
                            continue;
                        }
                        if (idx[i].length < result.length) {
                            result.length = idx[i].length;
                        }
                        for (let j = 0; j < result.length; ++j) {
                            result[j] = result[j].intersection(idx[i][j]);
                        }
                    }
                    return result;
                };
                /**
                 * Fetch a bitmap of potentially-matching functions,
                 * plus a list of query elements annotated with the correct IDs.
                 *
                 * More than one ID can exist because, for example, q=`Iter` can match
                 * `std::vec::Iter`, or `std::btree_set::Iter`, or anything else, and those
                 * items different IDs. What's worse, q=`Iter<Iter>` has N**2 possible
                 * matches, because it could be `vec::Iter<btree_set::Iter>`,
                 * `btree_set::Iter<vec::Iter>`, `vec::Iter<vec::Iter>`,
                 * `btree_set::Iter<btree_set::Iter>`,
                 * or anything else. This function returns all possible permutations.
                 *
                 * @param {rustdoc.ParserQueryElement|null} elem
                 * @param {rustdoc.TypeInvertedIndexPolarity} polarity
                 * @returns {Promise<PostingsList<rustdoc.QueryElement>[]>}
                 */
                const unpackPostingsList = async(elem, polarity) => {
                    if (!elem) {
                        return empty_postings_list;
                    }
                    const typeFilter = itemTypeFromName(elem.typeFilter);
                    const [searchResults, upla, uplb] = await Promise.all([
                        index.search(elem.normalizedPathLast),
                        unpackPostingsListAll(elem.generics, polarity),
                        unpackPostingsListBindings(elem.bindings, polarity),
                    ]);
                    /**
                     * @type {Promise<[
                     *     number,
                     *     string|null,
                     *     rustdoc.TypeData|null,
                     *     rustdoc.PathData|null,
                     * ]>[]}
                     * */
                    const typePromises = [];
                    if (typeFilter !== TY_GENERIC && searchResults) {
                        for (const id of searchResults.matches().entries()) {
                            typePromises.push(Promise.all([
                                this.getName(id),
                                this.getTypeData(id),
                                this.getPathData(id),
                            ]).then(([name, typeData, pathData]) =>
                                [id, name, typeData, pathData]));
                        }
                    }
                    const types = (await Promise.all(typePromises))
                        .filter(([_id, name, ty, path]) =>
                            name !== null && name.toLowerCase() === elem.pathLast &&
                            ty && !ty[polarity].every(bitmap => {
                                return bitmap.isEmpty();
                            }) &&
                            path && path.ty !== TY_ASSOCTYPE &&
                            (elem.pathWithoutLast.length === 0 ||
                                checkPath(
                                    elem.pathWithoutLast,
                                    path.modulePath.split("::"),
                                ) === 0),
                            );
                    if (types.length === 0) {
                        const areGenericsAllowed = typeFilter === TY_GENERIC || (
                            typeFilter === -1 &&
                            (parsedQuery.totalElems > 1 || parsedQuery.hasReturnArrow) &&
                            elem.pathWithoutLast.length === 0 &&
                            elem.generics.length === 0 &&
                            elem.bindings.size === 0
                        );
                        if (typeFilter !== TY_GENERIC &&
                            (elem.name.length >= 3 || !areGenericsAllowed)
                        ) {
                            /** @type {string|null} */
                            let chosenName = null;
                            /** @type {rustdoc.TypeData[]} */
                            let chosenType = [];
                            /** @type {rustdoc.PathData[]} */
                            let chosenPath = [];
                            /** @type {number[]} */
                            let chosenId = [];
                            let chosenDist = Number.MAX_SAFE_INTEGER;
                            const levResults = index.searchLev(elem.normalizedPathLast);
                            for await (const searchResults of levResults) {
                                for (const id of searchResults.matches().entries()) {
                                    const [name, ty, path] = await Promise.all([
                                        this.getName(id),
                                        this.getTypeData(id),
                                        this.getPathData(id),
                                    ]);
                                    if (name !== null && ty !== null && path !== null &&
                                        !ty[polarity].every(bitmap => {
                                            return bitmap.isEmpty();
                                        }) &&
                                        path.ty !== TY_ASSOCTYPE
                                    ) {
                                        let dist = editDistance(
                                            name,
                                            elem.pathLast,
                                            maxEditDistance,
                                        );
                                        if (elem.pathWithoutLast.length !== 0) {
                                            const pathDist = checkPath(
                                                elem.pathWithoutLast,
                                                path.modulePath.split("::"),
                                            );
                                            // guaranteed to be higher than the path limit
                                            dist += pathDist === null ?
                                                Number.MAX_SAFE_INTEGER :
                                                pathDist;
                                        }
                                        if (name === chosenName) {
                                            chosenId.push(id);
                                            chosenType.push(ty);
                                            chosenPath.push(path);
                                        } else if (dist < chosenDist) {
                                            chosenName = name;
                                            chosenId = [id];
                                            chosenType = [ty];
                                            chosenPath = [path];
                                            chosenDist = dist;
                                        }
                                    }
                                }
                                if (chosenId.length !== 0) {
                                    // searchLev returns results in order
                                    // if we have working matches, we're done
                                    break;
                                }
                            }
                            if (areGenericsAllowed) {
                                parsedQuery.proposeCorrectionFrom = elem.name;
                                parsedQuery.proposeCorrectionTo = chosenName;
                            } else {
                                parsedQuery.correction = chosenName;
                                for (let i = 0; i < chosenType.length; ++i) {
                                    types.push([
                                        chosenId[i],
                                        chosenName,
                                        chosenType[i],
                                        chosenPath[i],
                                    ]);
                                }
                            }
                        }
                        if (areGenericsAllowed) {
                            let genericId = genericMap.get(elem.normalizedPathLast);
                            if (genericId === undefined) {
                                genericId = genericMap.size;
                                genericMap.set(elem.normalizedPathLast, genericId);
                            }
                            return [{
                                invertedIndex: await this.getGenericInvertedIndex(genericId),
                                queryElem: {
                                    name: elem.name,
                                    id: (-genericId) - 1,
                                    typeFilter: TY_GENERIC,
                                    generics: [],
                                    bindings: EMPTY_BINDINGS_MAP,
                                    fullPath: elem.fullPath,
                                    pathLast: elem.pathLast,
                                    normalizedPathLast: elem.normalizedPathLast,
                                    pathWithoutLast: elem.pathWithoutLast,
                                },
                            }];
                        }
                    }
                    types.sort(([_i, name1, _t, pathData1], [_i2, name2, _t2, pathData2]) => {
                        const p1 = !pathData1 ? "" : pathData1.modulePath;
                        const p2 = !pathData2 ? "" : pathData2.modulePath;
                        const n1 = name1 === null ? "" : name1;
                        const n2 = name2 === null ? "" : name2;
                        if (p1.length !== p2.length) {
                            return p1.length > p2.length ? +1 : -1;
                        }
                        if (n1.length !== n2.length) {
                            return n1.length > n2.length ? +1 : -1;
                        }
                        if (n1 !== n2) {
                            return n1 > n2 ? +1 : -1;
                        }
                        if (p1 !== p2) {
                            return p1 > p2 ? +1 : -1;
                        }
                        return 0;
                    });
                    /** @type {PostingsList<rustdoc.QueryElement>[]} */
                    const results = [];
                    for (const [id, _name, typeData] of types) {
                        if (!typeData || typeData[polarity].every(bitmap => {
                            return bitmap.isEmpty();
                        })) {
                            continue;
                        }
                        for (const {invertedIndex: genericsIdx, queryElem: generics} of upla) {
                            for (const {invertedIndex: bindingsIdx, queryElem: bindings} of uplb) {
                                results.push({
                                    invertedIndex: intersectInvertedIndexes(
                                        typeData[polarity],
                                        genericsIdx,
                                        bindingsIdx,
                                    ),
                                    queryElem: {
                                        name: elem.name,
                                        id,
                                        typeFilter,
                                        generics,
                                        bindings,
                                        fullPath: elem.fullPath,
                                        pathLast: elem.pathLast,
                                        normalizedPathLast: elem.normalizedPathLast,
                                        pathWithoutLast: elem.pathWithoutLast,
                                    },
                                });
                                if ((results.length & 0x7F) === 0) {
                                    await yieldToBrowser();
                                }
                            }
                        }
                    }
                    return results;
                };
                /**
                 * Fetch all possible matching permutations of a list of query elements.
                 *
                 * The empty list returns an "identity postings list", with a bitmap that
                 * matches everything and an empty list of elems. This allows you to safely
                 * take the intersection of this bitmap.
                 *
                 * @param {(rustdoc.ParserQueryElement|null)[]|null} elems
                 * @param {rustdoc.TypeInvertedIndexPolarity} polarity
                 * @returns {Promise<PostingsList<rustdoc.QueryElement[]>[]>}
                 */
                const unpackPostingsListAll = async(elems, polarity) => {
                    if (!elems || elems.length === 0) {
                        return nested_everything_postings_list;
                    }
                    const [firstPostingsList, remainingAll] = await Promise.all([
                        unpackPostingsList(elems[0], polarity),
                        unpackPostingsListAll(elems.slice(1), polarity),
                    ]);
                    /** @type {PostingsList<rustdoc.QueryElement[]>[]} */
                    const results = [];
                    for (const {
                        invertedIndex: firstIdx,
                        queryElem: firstElem,
                    } of firstPostingsList) {
                        for (const {
                            invertedIndex: remainingIdx,
                            queryElem: remainingElems,
                        } of remainingAll) {
                            results.push({
                                invertedIndex: intersectInvertedIndexes(firstIdx, remainingIdx),
                                queryElem: [firstElem, ...remainingElems],
                            });
                            if ((results.length & 0x7F) === 0) {
                                await yieldToBrowser();
                            }
                        }
                    }
                    return results;
                };
                /**
                 * Fetch all possible matching permutations of a map query element bindings.
                 *
                 * The empty list returns an "identity postings list", with a bitmap that
                 * matches everything and an empty list of elems. This allows you to safely
                 * take the intersection of this bitmap.
                 *
                 * Heads up! This function mutates the Map that you provide.
                 * Before passing an actual parser item to it, make sure to clone the map.
                 *
                 * @param {Map<string, rustdoc.ParserQueryElement[]>} elems
                 * @param {rustdoc.TypeInvertedIndexPolarity} polarity
                 * @returns {Promise<PostingsList<
                 *     Map<number, rustdoc.QueryElement[]>,
                 * >[]>}
                 */
                const unpackPostingsListBindings = async(elems, polarity) => {
                    if (!elems) {
                        return [{
                            invertedIndex: everything_inverted_index,
                            queryElem: new Map(),
                        }];
                    }
                    const firstKey = elems.keys().next().value;
                    if (firstKey === undefined) {
                        return [{
                            invertedIndex: everything_inverted_index,
                            queryElem: new Map(),
                        }];
                    }
                    const firstList = elems.get(firstKey);
                    if (firstList === undefined) {
                        return [{
                            invertedIndex: everything_inverted_index,
                            queryElem: new Map(),
                        }];
                    }
                    // HEADS UP!
                    // We must put this map back the way we found it before returning,
                    // otherwise things break.
                    elems.delete(firstKey);
                    const [firstKeyIds, firstPostingsList, remainingAll] = await Promise.all([
                        index.search(firstKey),
                        unpackPostingsListAll(firstList, polarity),
                        unpackPostingsListBindings(elems, polarity),
                    ]);
                    if (!firstKeyIds) {
                        elems.set(firstKey, firstList);
                        // User specified a non-existent key.
                        return [{
                            invertedIndex: empty_inverted_index,
                            queryElem: new Map(),
                        }];
                    }
                    /** @type {PostingsList<Map<number, rustdoc.QueryElement[]>>[]} */
                    const results = [];
                    for (const keyId of firstKeyIds.matches().entries()) {
                        for (const {
                            invertedIndex: firstIdx,
                            queryElem: firstElem,
                        } of firstPostingsList) {
                            for (const {
                                invertedIndex: remainingIdx,
                                queryElem: remainingElems,
                            } of remainingAll) {
                                const elems = new Map(remainingElems);
                                elems.set(keyId, firstElem);
                                results.push({
                                    invertedIndex: intersectInvertedIndexes(firstIdx, remainingIdx),
                                    queryElem: elems,
                                });
                                if ((results.length & 0x7F) === 0) {
                                    await yieldToBrowser();
                                }
                            }
                        }
                    }
                    elems.set(firstKey, firstList);
                    if (results.length === 0) {
                        // User specified a non-existent key.
                        return [{
                            invertedIndex: empty_inverted_index,
                            queryElem: new Map(),
                        }];
                    }
                    return results;
                };

                // finally, we can do the actual unification loop
                const [allInputs, allOutput] = await Promise.all([
                    unpackPostingsListAll(inputs, "invertedFunctionInputsIndex"),
                    unpackPostingsListAll(output, "invertedFunctionOutputIndex"),
                ]);
                let checkCounter = 0;
                /**
                 * Finally, we can perform an incremental search, sorted by the number of
                 * entries that match a given query.
                 *
                 * The outer list gives the number of elements. The inner one is separate
                 * for each distinct name resolution.
                 *
                 * @type {{
                 *     bitmap: stringdex.RoaringBitmap,
                 *     inputs: rustdoc.QueryElement[],
                 *     output: rustdoc.QueryElement[],
                 * }[][]}
                 */
                const queryPlan = [];
                for (const {invertedIndex: inputsIdx, queryElem: inputs} of allInputs) {
                    for (const {invertedIndex: outputIdx, queryElem: output} of allOutput) {
                        const invertedIndex = intersectInvertedIndexes(inputsIdx, outputIdx);
                        for (const [size, bitmap] of invertedIndex.entries()) {
                            checkCounter += 1;
                            if ((checkCounter & 0x7F) === 0) {
                                await yieldToBrowser();
                            }
                            if (!queryPlan[size]) {
                                queryPlan[size] = [];
                            }
                            queryPlan[size].push({
                                bitmap, inputs, output,
                            });
                        }
                    }
                }
                const resultPromises = [];
                const dedup = new Set();
                let resultCounter = 0;
                const isReturnTypeQuery = inputs.length === 0;
                /** @type {rustdoc.PlainResultObject[]} */
                const pushToBottom = [];
                plan: for (const queryStep of queryPlan) {
                    for (const {bitmap, inputs, output} of queryStep) {
                        for (const id of bitmap.entries()) {
                            checkCounter += 1;
                            if ((checkCounter & 0x7F) === 0) {
                                await yieldToBrowser();
                            }
                            resultPromises.push(this.getFunctionData(id).then(async fnData => {
                                if (!fnData || !fnData.functionSignature) {
                                    return null;
                                }
                                checkCounter += 1;
                                if ((checkCounter & 0x7F) === 0) {
                                    await yieldToBrowser();
                                }
                                const functionSignature = fnData.functionSignature;
                                if (!unifyFunctionTypes(
                                    functionSignature.inputs,
                                    inputs,
                                    functionSignature.where_clause,
                                    null,
                                    mgens => {
                                        return !!unifyFunctionTypes(
                                            functionSignature.output,
                                            output,
                                            functionSignature.where_clause,
                                            mgens,
                                            checkTypeMgensForConflict,
                                            0, // unboxing depth
                                        );
                                    },
                                    0, // unboxing depth
                                )) {
                                    return null;
                                }
                                const result = {
                                    id,
                                    dist: fnData.elemCount,
                                    path_dist: 0,
                                    index: -1,
                                    elems: inputs,
                                    returned: output,
                                    is_alias: false,
                                };
                                const entry = await this.getEntryData(id);
                                if ((entry && !isFnLikeTy(entry.ty)) ||
                                    (isReturnTypeQuery &&
                                        functionSignature &&
                                        containsTypeFromQuery(
                                            output,
                                            functionSignature.inputs,
                                            functionSignature.where_clause,
                                        )
                                    )
                                ) {
                                    pushToBottom.push(result);
                                    return null;
                                }
                                return result;
                            }));
                        }
                    }
                    for await (const result of sortAndTransformResults(
                        await Promise.all(resultPromises),
                        typeInfo,
                        currentCrate,
                        dedup,
                    )) {
                        if (resultCounter >= MAX_RESULTS) {
                            break plan;
                        }
                        yield result;
                        resultCounter += 1;
                    }
                    resultPromises.length = 0;
                }
                if (resultCounter >= MAX_RESULTS) {
                    return;
                }
                for await (const result of sortAndTransformResults(
                    await Promise.all(pushToBottom),
                    typeInfo,
                    currentCrate,
                    dedup,
                )) {
                    if (resultCounter >= MAX_RESULTS) {
                        break;
                    }
                    yield result;
                    resultCounter += 1;
                }
            }
            .bind(this);

        if (parsedQuery.foundElems === 1 && !parsedQuery.hasReturnArrow) {
            // We never want the main tab to delay behind the other two tabs.
            // This is a bit of a hack (because JS's scheduler doesn't have much of an API),
            // along with making innerRunTypeQuery yield to the UI thread.
            const {
                promise: donePromise,
                resolve: doneResolve,
                reject: doneReject,
            } = Promise.withResolvers();
            const doneTimeout = timeout(250);
            return {
                "in_args": (async function*() {
                    await Promise.race([donePromise, doneTimeout]);
                    yield* innerRunTypeQuery(parsedQuery.elems, [], "elems", currentCrate);
                })(),
                "returned": (async function*() {
                    await Promise.race([donePromise, doneTimeout]);
                    yield* innerRunTypeQuery([], parsedQuery.elems, "returned", currentCrate);
                })(),
                "others": (async function*() {
                    try {
                        yield* innerRunNameQuery(currentCrate);
                        doneResolve(null);
                    } catch (e) {
                        doneReject(e);
                        throw e;
                    }
                })(),
                "query": parsedQuery,
            };
        } else if (parsedQuery.error !== null) {
            return {
                "in_args": (async function*() {})(),
                "returned": (async function*() {})(),
                "others": innerRunNameQuery(currentCrate),
                "query": parsedQuery,
            };
        } else {
            const typeInfo = parsedQuery.elems.length === 0 ?
                "returned" : (
                    parsedQuery.returned.length === 0 ? "elems" : "sig"
                );
            return {
                "in_args": (async function*() {})(),
                "returned": (async function*() {})(),
                "others": parsedQuery.foundElems === 0 ?
                    (async function*() {})() :
                    innerRunTypeQuery(
                        parsedQuery.elems,
                        parsedQuery.returned,
                        typeInfo,
                        currentCrate,
                    ),
                "query": parsedQuery,
            };
        }
    }
}


// ==================== Core search logic end ====================

/** @type {DocSearch} */
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
    "",
    "attribute",
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

    // @ts-expect-error
    if (elem && elem.value !== "all crates") {
        // @ts-expect-error
        return elem.value;
    }
    return null;
}

// @ts-expect-error
function nextTab(direction) {
    const next = (searchState.currentTab + direction + 3) % searchState.focusedByTab.length;
    window.searchState.focusedByTab[searchState.currentTab] = document.activeElement;
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
    if (target && target instanceof HTMLElement) {
        target.focus();
    }
}

/**
 * Render a set of search results for a single tab.
 * @param {AsyncGenerator<rustdoc.ResultObject>} results   - The search results for this tab
 * @param {rustdoc.ParsedQuery<rustdoc.ParserQueryElement>} query
 * @param {boolean}     display - True if this is the active tab
 * @param {function(number, HTMLElement): any} finishedCallback
 * @param {boolean} isTypeSearch
 * @returns {Promise<HTMLElement>}
 */
async function addTab(results, query, display, finishedCallback, isTypeSearch) {
    const extraClass = display ? " active" : "";

    /** @type {HTMLElement} */
    let output = document.createElement("ul");
    output.className = "search-results " + extraClass;

    let count = 0;

    /** @type {Promise<string|null>[]} */
    const descList = [];

    /** @param {rustdoc.ResultObject} obj */
    const addNextResultToOutput = async obj => {
        count += 1;

        const name = obj.item.name;
        const type = itemTypes[obj.item.ty];
        const longType = longItemTypes[obj.item.ty];
        const typeName = longType.length !== 0 ? `${longType}` : "?";

        const link = document.createElement("a");
        link.className = "result-" + type;
        link.href = obj.href;

        const resultName = document.createElement("span");
        resultName.className = "result-name";

        resultName.insertAdjacentHTML(
            "beforeend",
            `<span class="typename">${typeName}</span>`);
        link.appendChild(resultName);

        let alias = " ";
        if (obj.alias !== undefined) {
            alias = ` <div class="alias">\
<b>${obj.alias}</b><i class="grey">&nbsp;- see&nbsp;</i>\
</div>`;
        }
        resultName.insertAdjacentHTML(
            "beforeend",
            `<div class="path">${alias}\
${obj.displayPath}<span class="${type}">${name}</span>\
</div>`);

        const description = document.createElement("div");
        description.className = "desc";
        obj.desc.then(desc => {
            if (desc !== null) {
                description.insertAdjacentHTML("beforeend", desc);
            }
        });
        descList.push(obj.desc);
        if (obj.displayTypeSignature) {
            const {type, mappedNames, whereClause} = await obj.displayTypeSignature;
            const displayType = document.createElement("div");
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
        output.appendChild(link);

        results.next().then(async nextResult => {
            if (nextResult.value) {
                addNextResultToOutput(nextResult.value);
            } else {
                await Promise.all(descList);
                // need to make sure the element is shown before
                // running this callback
                yieldToBrowser().then(() => finishedCallback(count, output));
            }
        });
    };
    const firstResult = await results.next();
    let correctionOutput = "";
    if (query.correction !== null && isTypeSearch) {
        const orig = query.returned.length > 0
            ? query.returned[0].name
            : query.elems[0].name;
        correctionOutput = "<h3 class=\"search-corrections\">" +
            `Type "${orig}" not found. ` +
            "Showing results for closest type name " +
            `"${query.correction}" instead.</h3>`;
    }
    if (query.proposeCorrectionFrom !== null && isTypeSearch) {
        const orig = query.proposeCorrectionFrom;
        const targ = query.proposeCorrectionTo;
        correctionOutput = "<h3 class=\"search-corrections\">" +
            `Type "${orig}" not found and used as generic parameter. ` +
            `Consider searching for "${targ}" instead.</h3>`;
    }
    if (firstResult.value) {
        if (correctionOutput !== "") {
            const h3 = document.createElement("h3");
            h3.innerHTML = correctionOutput;
            output.appendChild(h3);
        }
        await addNextResultToOutput(firstResult.value);
    } else {
        output = document.createElement("div");
        if (correctionOutput !== "") {
            const h3 = document.createElement("h3");
            h3.innerHTML = correctionOutput;
            output.appendChild(h3);
        }
        output.className = "search-failed" + extraClass;
        const dlroChannel = `https://doc.rust-lang.org/${getVar("channel")}`;
        if (query.userQuery !== "") {
            output.innerHTML += "No results :(<br/>" +
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
        output.innerHTML += "Example searches:<ul>" +
            "<li><a href=\"" + getNakedUrl() + "?search=std::vec\">std::vec</a></li>" +
            "<li><a href=\"" + getNakedUrl() + "?search=u32+->+bool\">u32 -> bool</a></li>" +
            "<li><a href=\"" + getNakedUrl() + "?search=Option<T>,+(T+->+U)+->+Option<U>\">" +
                "Option&lt;T>, (T -> U) -> Option&lt;U></a></li>" +
            "</ul>";
        // need to make sure the element is shown before
        // running this callback
        yieldToBrowser().then(() => finishedCallback(0, output));
    }
    return output;
}

/**
 * returns [tab, output]
 * @param {number} tabNb
 * @param {string} text
 * @param {AsyncGenerator<rustdoc.ResultObject>} results
 * @param {rustdoc.ParsedQuery<rustdoc.ParserQueryElement>} query
 * @param {boolean} isTypeSearch
 * @param {boolean} goToFirst
 * @returns {[HTMLElement, Promise<HTMLElement>]}
 */
function makeTab(tabNb, text, results, query, isTypeSearch, goToFirst) {
    const isCurrentTab = window.searchState.currentTab === tabNb;
    const tabButton = document.createElement("button");
    tabButton.appendChild(document.createTextNode(text));
    tabButton.className = isCurrentTab ? "selected" : "";
    const tabCount = document.createElement("span");
    tabCount.className = "count loading";
    tabCount.innerHTML = "\u{2007}(\u{2007})\u{2007}\u{2007}";
    tabButton.appendChild(tabCount);
    return [
        tabButton,
        addTab(results, query, isCurrentTab, (count, output) => {
            const search = window.searchState.outputElement();
            const error = query.error;
            if (count === 0 && error !== null && search) {
                error.forEach((value, index) => {
                    value = value.split("<").join("&lt;").split(">").join("&gt;");
                    if (index % 2 !== 0) {
                        error[index] = `<code>${value.replaceAll(" ", "&nbsp;")}</code>`;
                    } else {
                        error[index] = value;
                    }
                });
                const errorReport = document.createElement("h3");
                errorReport.className = "error";
                errorReport.innerHTML = `Query parser error: "${error.join("")}".`;
                search.insertBefore(errorReport, search.firstElementChild);
            } else if (goToFirst ||
                (count === 1 && getSettingValue("go-to-only-result") === "true")
            ) {
                // Needed to force re-execution of JS when coming back to a page. Let's take this
                // scenario as example:
                //
                // 1. You have the "Directly go to item in search if there is only one result"
                //    option enabled.
                // 2. You make a search which results only one result, leading you automatically to
                //    this result.
                // 3. You go back to previous page.
                //
                // Now, without the call below, the JS will not be re-executed and the previous
                // state will be used, starting search again since the search input is not empty,
                // leading you back to the previous page again.
                window.onunload = () => { };
                window.searchState.removeQueryParameters();
                const a = output.querySelector("a");
                if (a) {
                    a.click();
                    return;
                }
            }

            // https://blog.horizon-eda.org/misc/2020/02/19/ui.html
            //
            // CSS runs with `font-variant-numeric: tabular-nums` to ensure all
            // digits are the same width. \u{2007} is a Unicode space character
            // that is defined to be the same width as a digit.
            const fmtNbElems =
                count < 10  ? `\u{2007}(${count})\u{2007}\u{2007}` :
                count < 100 ? `\u{2007}(${count})\u{2007}` : `\u{2007}(${count})`;
            tabCount.innerHTML = fmtNbElems;
            tabCount.className = "count";
        }, isTypeSearch),
    ];
}

/**
 * @param {DocSearch} docSearch
 * @param {rustdoc.ResultsTable} results
 * @param {boolean} goToFirst
 * @param {string} filterCrates
 */
async function showResults(docSearch, results, goToFirst, filterCrates) {
    const search = window.searchState.outputElement();

    if (!search) {
        return;
    }

    let crates = "";
    const crateNames = await docSearch.getCrateNameList();
    if (crateNames.length > 1) {
        crates = "&nbsp;in&nbsp;<div id=\"crate-search-div\">" +
            "<select id=\"crate-search\"><option value=\"all crates\">all crates</option>";
        const l = crateNames.length;
        for (let i = 0; i < l; i += 1) {
            const c = crateNames[i];
            crates += `<option value="${c}" ${c === filterCrates && "selected"}>${c}</option>`;
        }
        crates += "</select></div>";
    }
    nonnull(document.querySelector(".search-switcher")).innerHTML = `Search results${crates}`;

    /** @type {[HTMLElement, Promise<HTMLElement>][]} */
    const tabs = [];
    searchState.currentTab = 0;
    if (results.query.error !== null) {
        tabs.push(makeTab(0, "In Names", results.others, results.query, false, goToFirst));
    } else if (
        results.query.foundElems <= 1 &&
        results.query.returned.length === 0 &&
        !results.query.hasReturnArrow
    ) {
        tabs.push(makeTab(0, "In Names", results.others, results.query, false, goToFirst));
        tabs.push(makeTab(1, "In Parameters", results.in_args, results.query, true, false));
        tabs.push(makeTab(2, "In Return Types", results.returned, results.query, true, false));
    } else {
        const signatureTabTitle =
            results.query.elems.length === 0 ? "In Function Return Types" :
                results.query.returned.length === 0 ? "In Function Parameters" :
                    "In Function Signatures";
        tabs.push(makeTab(0, signatureTabTitle, results.others, results.query, true, goToFirst));
    }

    const tabsElem = document.createElement("div");
    tabsElem.id = "search-tabs";

    const resultsElem = document.createElement("div");
    resultsElem.id = "results";

    search.innerHTML = "";
    for (const [tab, output] of tabs) {
        tabsElem.appendChild(tab);
        const placeholder = document.createElement("div");
        output.then(output => {
            if (placeholder.parentElement) {
                placeholder.parentElement.replaceChild(output, placeholder);
            }
        });
        resultsElem.appendChild(placeholder);
    }

    if (window.searchState.rustdocToolbar) {
        nonnull(
            nonnull(window.searchState.containerElement())
                .querySelector(".main-heading"),
        ).appendChild(window.searchState.rustdocToolbar);
    }
    const crateSearch = document.getElementById("crate-search");
    if (crateSearch) {
        crateSearch.addEventListener("input", updateCrate);
    }
    search.appendChild(tabsElem);
    search.appendChild(resultsElem);
    // Reset focused elements.
    window.searchState.showResults();
    window.searchState.focusedByTab = [null, null, null];
    let i = 0;
    for (const elem of tabsElem.childNodes) {
        const j = i;
        // @ts-expect-error
        elem.onclick = () => printTab(j);
        window.searchState.focusedByTab[i] = null;
        i += 1;
    }
    printTab(0);
}

// @ts-expect-error
function updateSearchHistory(url) {
    const btn = document.querySelector("#search-button a");
    if (btn instanceof HTMLAnchorElement) {
        btn.href = url;
    }
    if (!browserSupportsHistoryApi()) {
        return;
    }
    const params = searchState.getQueryStringParams();
    if (!history.state && params.search === undefined) {
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
    const query = DocSearch.parseQuery(nonnull(window.searchState.inputElement()).value.trim());

    let filterCrates = getFilterCrates();

    // @ts-expect-error
    if (!forced && query.userQuery === currentResults) {
        if (query.userQuery.length > 0) {
            putBackSearch();
        }
        return;
    }
    currentResults = query.userQuery;

    searchState.setLoadingSearch();

    const params = searchState.getQueryStringParams();

    // In case we have no information about the saved crate and there is a URL query parameter,
    // we override it with the URL query parameter.
    if (filterCrates === null && params["filter-crate"] !== undefined) {
        filterCrates = params["filter-crate"];
    }

    if (filterCrates !== null &&
        (await docSearch.getCrateNameList()).indexOf(filterCrates) === -1
    ) {
        filterCrates = null;
    }

    // Update document title to maintain a meaningful browser history
    searchState.title = "\"" + query.userQuery + "\" Search - Rust";

    // Because searching is incremental by character, only the most
    // recent search query is added to the browser history.
    updateSearchHistory(buildUrl(query.userQuery, filterCrates));

    await showResults(
        docSearch,
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
    const search_input = window.searchState.inputElement();
    if (!search_input) {
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
    const inputElement = nonnull(window.searchState.inputElement());
    if (inputElement.value === "") {
        inputElement.value = params.search || "";
    }

    const searchAfter500ms = () => {
        searchState.clearInputTimeout();
        window.searchState.timeout = setTimeout(search, 500);
    };
    inputElement.onkeyup = searchAfter500ms;
    inputElement.oninput = searchAfter500ms;
    if (inputElement.form) {
        inputElement.form.onsubmit = onSearchSubmit;
    }
    inputElement.onchange = e => {
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
    inputElement.onpaste = inputElement.onchange;

    // @ts-expect-error
    searchState.outputElement().addEventListener("keydown", e => {
        if (!(e instanceof KeyboardEvent)) {
            return;
        }
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

    inputElement.addEventListener("keydown", e => {
        if (e.which === 40) { // down
            focusSearchResult();
            e.preventDefault();
        }
    });

    inputElement.addEventListener("focus", () => {
        putBackSearch();
    });
}

// @ts-expect-error
function updateCrate(ev) {
    if (ev.target.value === "all crates") {
        // If we don't remove it from the URL, it'll be picked up again by the search.
        const query = nonnull(window.searchState.inputElement()).value.trim();
        updateSearchHistory(buildUrl(query, null));
    }
    // In case you "cut" the entry from the search input, then change the crate filter
    // before paste back the previous search, you get the old search results without
    // the filter. To prevent this, we need to remove the previous results.
    currentResults = null;
    search(true);
}

// eslint-disable-next-line max-len
// polyfill https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Uint8Array/fromBase64
/**
 * @type {function(string): Uint8Array} base64
 */
//@ts-expect-error
const makeUint8ArrayFromBase64 = Uint8Array.fromBase64 ? Uint8Array.fromBase64 : (string => {
    const bytes_as_string = atob(string);
    const l = bytes_as_string.length;
    const bytes = new Uint8Array(l);
    for (let i = 0; i < l; ++i) {
        bytes[i] = bytes_as_string.charCodeAt(i);
    }
    return bytes;
});


if (ROOT_PATH === null) {
    return;
}
const database = await Stringdex.loadDatabase(hooks);
if (typeof window !== "undefined") {
    docSearch = new DocSearch(ROOT_PATH, database);
    await docSearch.buildIndex();
    onEachLazy(document.querySelectorAll(
        ".search-form.loading",
    ), form => {
        removeClass(form, "loading");
    });
    registerSearchEvents();
    // If there's a search term in the URL, execute the search now.
    if (window.searchState.getQueryStringParams().search !== undefined) {
        search();
    }
} else if (typeof exports !== "undefined") {
    docSearch = new DocSearch(ROOT_PATH, database);
    await docSearch.buildIndex();
    return { docSearch, DocSearch };
}
};

if (typeof window !== "undefined") {
    const ROOT_PATH = window.rootPath;
    /** @type {stringdex.Callbacks|null} */
    let databaseCallbacks = null;
    initSearch(window.Stringdex, window.RoaringBitmap, {
        loadRoot: callbacks => {
            for (const key in callbacks) {
                if (Object.hasOwn(callbacks, key)) {
                    // @ts-ignore
                    window[key] = callbacks[key];
                }
            }
            databaseCallbacks = callbacks;
            // search.index/root is loaded by main.js, so
            // this script doesn't need to launch it, but
            // must pick it up
            if (window.searchIndex) {
                window.rr_(window.searchIndex);
            }
        },
        loadTreeByHash: hashHex => {
            const script = document.createElement("script");
            script.src = `${ROOT_PATH}search.index/${hashHex}.js`;
            script.onerror = e => {
                if (databaseCallbacks) {
                    databaseCallbacks.err_rn_(hashHex, e);
                }
            };
            document.documentElement.appendChild(script);
        },
        loadDataByNameAndHash: (name, hashHex) => {
            const script = document.createElement("script");
            script.src = `${ROOT_PATH}search.index/${name}/${hashHex}.js`;
            script.onerror = e => {
                if (databaseCallbacks) {
                    databaseCallbacks.err_rd_(hashHex, e);
                }
            };
            document.documentElement.appendChild(script);
        },
    });
} else if (typeof exports !== "undefined") {
    // eslint-disable-next-line no-undef
    exports.initSearch = initSearch;
}
