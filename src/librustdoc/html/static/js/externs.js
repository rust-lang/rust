// This file contains type definitions that are processed by the Closure Compiler but are
// not put into the JavaScript we include as part of the documentation. It is used for
// type checking. See README.md in this directory for more info.

/* eslint-disable */
var searchState;
function initSearch(searchIndex){}

/**
 * @typedef {{
 *     isExact: boolean,
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
 *     original: string,
 *     userQuery: string,
 *     length: number,
 *     pos: number,
 *     typeFilter: number,
 *     elems: Array<QueryElement>,
 *     elemName: (string|null),
 *     args: Array<QueryElement>,
 *     returned: Array<QueryElement>,
 *     foundElems: number,
 *     id: string,
 *     nameSplit: (string|null),
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
