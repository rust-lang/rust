// This file contains type definitions that are processed by the Closure Compiler but are
// not put into the JavaScript we include as part of the documentation. It is used for
// type checking. See README.md in this directory for more info.
/* eslint-env es6 */
/* eslint no-var: "error" */
/* eslint prefer-const: "error" */

/* eslint-disable */
let searchState;
function initSearch(searchIndex){}

/**
 * @typedef {{
 *     name: string,
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
 *    type: (Array<?>|null)
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
let Results;
