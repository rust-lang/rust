// This file contains type definitions that are processed by the Closure Compiler but are
// not put into the JavaScript we include as part of the documentation. It is used for
// type checking. See README.md in this directory for more info.

/* eslint-disable */
var searchState;
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
var QueryElement;

/**
 * @typedef {{
 *      pos: number,
 *      totalElems: number,
 *      typeFilter: (null|string),
 *      userQuery: string,
 * }}
 */
var ParserState;

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
var ParsedQuery;

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
var Row;

/**
 * @typedef {{
 *    in_args: Array<Object>,
 *    returned: Array<Object>,
 *    others: Array<Object>,
 *    query: ParsedQuery,
 * }}
 */
var ResultsTable;

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
var Results;
