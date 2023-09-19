// This file contains type definitions that are processed by the Closure Compiler but are
// not put into the JavaScript we include as part of the documentation. It is used for
// type checking. See README.md in this directory for more info.

/* eslint-disable */
let searchState;
function initSearch(searchIndex){}

/**
 * @typedef {{
 *     name: string,
 *     id: integer|null,
 *     fullPath: Array<string>,
 *     pathWithoutLast: Array<string>,
 *     pathLast: string,
 *     generics: Array<QueryElement>,
 * }}
 */
let QueryElement;

/**
 * @typedef {{
 *      pos: number,
 *      totalElems: number,
 *      typeFilter: (null|string),
 *      userQuery: string,
 * }}
 */
let ParserState;

/**
 * @typedef {{
 *     original: string,
 *     userQuery: string,
 *     typeFilter: number,
 *     elems: Array<QueryElement>,
 *     args: Array<QueryElement>,
 *     returned: Array<QueryElement>,
 *     foundElems: number,
 *     totalElems: number,
 *     literalSearch: boolean,
 *     corrections: Array<{from: string, to: integer}>,
 * }}
 */
let ParsedQuery;

/**
 * @typedef {{
 *    crate: string,
 *    desc: string,
 *    id: number,
 *    name: string,
 *    normalizedName: string,
 *    parent: (Object|null|undefined),
 *    path: string,
 *    ty: (Number|null|number),
 *    type: FunctionSearchType?
 * }}
 */
let Row;

/**
 * @typedef {{
 *    in_args: Array<Object>,
 *    returned: Array<Object>,
 *    others: Array<Object>,
 *    query: ParsedQuery,
 * }}
 */
let ResultsTable;

/**
 * @typedef {Map<String, ResultObject>}
 */
let Results;

/**
 * @typedef {{
 *     desc: string,
 *     displayPath: string,
 *     fullPath: string,
 *     href: string,
 *     id: number,
 *     lev: number,
 *     name: string,
 *     normalizedName: string,
 *     parent: (Object|undefined),
 *     path: string,
 *     ty: number,
 * }}
 */
let ResultObject;

/**
 * A pair of [inputs, outputs], or 0 for null. This is stored in the search index.
 * The JavaScript deserializes this into FunctionSearchType.
 *
 * Numeric IDs are *ONE-indexed* into the paths array (`p`). Zero is used as a sentinel for `null`
 * because `null` is four bytes while `0` is one byte.
 *
 * An input or output can be encoded as just a number if there is only one of them, AND
 * it has no generics. The no generics rule exists to avoid ambiguity: imagine if you had
 * a function with a single output, and that output had a single generic:
 *
 *     fn something() -> Result<usize, usize>
 *
 * If output was allowed to be any RawFunctionType, it would look like thi
 *
 *     [[], [50, [3, 3]]]
 *
 * The problem is that the above output could be interpreted as either a type with ID 50 and two
 * generics, or it could be interpreted as a pair of types, the first one with ID 50 and the second
 * with ID 3 and a single generic parameter that is also ID 3. We avoid this ambiguity by choosing
 * in favor of the pair of types interpretation. This is why the `(number|Array<RawFunctionType>)`
 * is used instead of `(RawFunctionType|Array<RawFunctionType>)`.
 *
 * The output can be skipped if it's actually unit and there's no type constraints. If thi
 * function accepts constrained generics, then the output will be unconditionally emitted, and
 * after it will come a list of trait constraints. The position of the item in the list will
 * determine which type parameter it is. For example:
 *
 *     [1, 2, 3, 4, 5]
 *      ^  ^  ^  ^  ^
 *      |  |  |  |  - generic parameter (-3) of trait 5
 *      |  |  |  - generic parameter (-2) of trait 4
 *      |  |  - generic parameter (-1) of trait 3
 *      |  - this function returns a single value (type 2)
 *      - this function takes a single input parameter (type 1)
 *
 * Or, for a less contrived version:
 *
 *     [[[4, -1], 3], [[5, -1]], 11]
 *      -^^^^^^^----   ^^^^^^^   ^^
 *       |        |    |          - generic parameter, roughly `where -1: 11`
 *       |        |    |            since -1 is the type parameter and 11 the trait
 *       |        |    - function output 5<-1>
 *       |        - the overall function signature is something like
 *       |          `fn(4<-1>, 3) -> 5<-1> where -1: 11`
 *       - function input, corresponds roughly to 4<-1>
 *         4 is an index into the `p` array for a type
 *         -1 is the generic parameter, given by 11
 *
 * If a generic parameter has multiple trait constraints, it gets wrapped in an array, just like
 * function inputs and outputs:
 *
 *     [-1, -1, [4, 3]]
 *              ^^^^^^ where -1: 4 + 3
 *
 * If a generic parameter's trait constraint has generic parameters, it gets wrapped in the array
 * even if only one exists. In other words, the ambiguity of `4<3>` and `4 + 3` is resolved in
 * favor of `4 + 3`:
 *
 *     [-1, -1, [[4, 3]]]
 *              ^^^^^^^^ where -1: 4 + 3
 *
 *     [-1, -1, [5, [4, 3]]]
 *              ^^^^^^^^^^^ where -1: 5, -2: 4 + 3
 *
 * If a generic parameter has no trait constraints (like in Rust, the `Sized` constraint i
 * implied and a fake `?Sized` constraint used to note its absence), it will be filled in with 0.
 *
 * @typedef {(
 *     0 |
 *     [(number|Array<RawFunctionType>)] |
 *     [(number|Array<RawFunctionType>), (number|Array<RawFunctionType>)] |
 *     Array<(number|Array<RawFunctionType>)>
 * )}
 */
let RawFunctionSearchType;

/**
 * A single function input or output type. This is either a single path ID, or a pair of
 * [path ID, generics].
 *
 * Numeric IDs are *ONE-indexed* into the paths array (`p`). Zero is used as a sentinel for `null`
 * because `null` is four bytes while `0` is one byte.
 *
 * @typedef {number | [number, Array<RawFunctionType>]}
 */
let RawFunctionType;

/**
 * @typedef {{
 *     inputs: Array<FunctionType>,
 *     output: Array<FunctionType>,
 *     where_clause: Array<Array<FunctionType>>,
 * }}
 */
let FunctionSearchType;

/**
 * @typedef {{
 *     id: (null|number),
 *     ty: (null|number),
 *     generics: Array<FunctionType>,
 * }}
 */
let FunctionType;
