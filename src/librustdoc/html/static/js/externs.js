// This file contains type definitions that are processed by the Closure Compiler but are
// not put into the JavaScript we include as part of the documentation. It is used for
// type checking. See README.md in this directory for more info.

/* eslint-disable */
var searchState;
function initSearch(searchIndex){}

/**
 * @typedef {{
 *   raw: string,
 *   query: string,
 *   type: string,
 *   id: string,
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
