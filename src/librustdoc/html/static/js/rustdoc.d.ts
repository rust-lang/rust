// This file contains type definitions that are processed by the TypeScript Compiler but are
// not put into the JavaScript we include as part of the documentation. It is used for
// type checking. See README.md in this directory for more info.

/* eslint-disable */
declare global {
    /** Map from crate name to directory structure, for source view */
    declare var srcIndex: Map<string, rustdoc.Dir>;
    /** Defined and documented in `storage.js` */
    declare function nonnull(x: T|null, msg: string|undefined);
    /** Defined and documented in `storage.js` */
    declare function nonundef(x: T|undefined, msg: string|undefined);
    interface Window {
        /** Make the current theme easy to find */
        currentTheme: HTMLLinkElement|null;
        /** Generated in `render/context.rs` */
        SIDEBAR_ITEMS?: { [key: string]: string[] };
        /** Used by the popover tooltip code. */
        RUSTDOC_TOOLTIP_HOVER_MS: number;
        /** Used by the popover tooltip code. */
        RUSTDOC_TOOLTIP_HOVER_EXIT_MS: number;
        /** Search engine data used by main.js and search.js */
        searchState: rustdoc.SearchState;
        /** Global option, with a long list of "../"'s */
        rootPath: string|null;
        /**
         * Currently opened crate.
         * As a multi-page application, we know this never changes once set.
         */
        currentCrate: string|null;
        /**
         * Hide popovers, tooltips, or the mobile sidebar.
         *
         * Pass `true` to reset focus for tooltip popovers.
         */
        hideAllModals: function(boolean),
        /**
         * Hide popovers, but leave other modals alone.
         */
        hidePopoverMenus: function(),
        /**
         * Hide the source page sidebar. If it's already closed,
         * or if this is a docs page, this function does nothing.
         */
        rustdocCloseSourceSidebar: function(),
        /**
         * Show the source page sidebar. If it's already opened,
         * or if this is a docs page, this function does nothing.
         */
        rustdocShowSourceSidebar: function(),
        /**
         * Close the sidebar in source code view
         */
        rustdocCloseSourceSidebar?: function(),
        /**
         * Shows the sidebar in source code view
         */
        rustdocShowSourceSidebar?: function(),
        /**
         * Toggles the sidebar in source code view
         */
        rustdocToggleSrcSidebar?: function(),
        /**
         * create's the sidebar in source code view.
         * called in generated `src-files.js`.
         */
        createSrcSidebar?: function(),
        /**
         * Set up event listeners for a scraped source example.
         */
        updateScrapedExample?: function(HTMLElement, HTMLElement),
        /**
         * register trait implementors, called by code generated in
         * `write_shared.rs`
         */
        register_implementors?: function(rustdoc.Implementors): void,
        /**
         * fallback in case `register_implementors` isn't defined yet.
         */
        pending_implementors?: rustdoc.Implementors,
        register_type_impls?: function(rustdoc.TypeImpls): void,
        pending_type_impls?: rustdoc.TypeImpls,
        rustdoc_add_line_numbers_to_examples?: function(),
        rustdoc_remove_line_numbers_from_examples?: function(),
    }
    interface HTMLElement {
        /** Used by the popover tooltip code. */
        TOOLTIP_FORCE_VISIBLE: boolean|undefined,
        /** Used by the popover tooltip code */
        TOOLTIP_HOVER_TIMEOUT: Timeout|undefined,
    }
}

export = rustdoc;

declare namespace rustdoc {
    interface SearchState {
        rustdocToolbar: HTMLElement|null;
        loadingText: string;
        input: HTMLInputElement|null;
        title: string;
        titleBeforeSearch: string;
        timeout: number|null;
        currentTab: number;
        focusedByTab: [number|null, number|null, number|null];
        clearInputTimeout: function;
        outputElement: function(): HTMLElement|null;
        focus: function();
        defocus: function();
        showResults: function(HTMLElement|null|undefined);
        removeQueryParameters: function();
        hideResults: function();
        getQueryStringParams: function(): Object.<any, string>;
        origPlaceholder: string;
        setup: function();
        setLoadingSearch: function();
        descShards: Map<string, SearchDescShard[]>;
        loadDesc: function({descShard: SearchDescShard, descIndex: number}): Promise<string|null>;
        loadedDescShard: function(string, number, string);
        isDisplayed: function(): boolean,
    }

    interface SearchDescShard {
        crate: string;
        promise: Promise<string[]>|null;
        resolve: function(string[])|null;
        shard: number;
    }

    /**
     * A single parsed "atom" in a search query. For example,
     * 
     *     std::fmt::Formatter, Write -> Result<()>
     *     ┏━━━━━━━━━━━━━━━━━━  ┌────    ┏━━━━━┅┅┅┅┄┄┄┄┄┄┄┄┄┄┄┄┄┄┐
     *     ┃                    │        ┗ QueryElement {        ┊
     *     ┃                    │              name: Result      ┊
     *     ┃                    │              generics: [       ┊
     *     ┃                    │                   QueryElement ┘
     *     ┃                    │                   name: ()
     *     ┃                    │              ]
     *     ┃                    │          }
     *     ┃                    └ QueryElement {
     *     ┃                          name: Write
     *     ┃                      }
     *     ┗ QueryElement {
     *           name: Formatter
     *           pathWithoutLast: std::fmt
     *       }
     */
    interface QueryElement {
        name: string,
        id: number|null,
        fullPath: Array<string>,
        pathWithoutLast: Array<string>,
        pathLast: string,
        normalizedPathLast: string,
        generics: Array<QueryElement>,
        bindings: Map<number, Array<QueryElement>>,
        typeFilter: number|null,
    }

    /**
     * Same as QueryElement, but bindings and typeFilter support strings
     */
    interface ParserQueryElement {
        name: string|null,
        id: number|null,
        fullPath: Array<string>,
        pathWithoutLast: Array<string>,
        pathLast: string,
        normalizedPathLast: string,
        generics: Array<ParserQueryElement>,
        bindings: Map<string, Array<ParserQueryElement>>,
        bindingName: {name: string|null, generics: ParserQueryElement[]}|null,
        typeFilter: number|string|null,
    }

    /**
     * Same as ParserQueryElement, but all fields are optional.
     */
    type ParserQueryElementFields = {
        [K in keyof ParserQueryElement]?: ParserQueryElement[T]
    }
    /**
     * Intermediate parser state. Discarded when parsing is done.
     */
    interface ParserState {
        pos: number;
        length: number;
        totalElems: number;
        genericsElems: number;
        typeFilter: (null|string);
        userQuery: string;
        isInBinding: (null|{name: string, generics: ParserQueryElement[]});
    }

    /**
     * A complete parsed query.
     */
    interface ParsedQuery<T> {
        userQuery: string,
        elems: Array<T>,
        returned: Array<T>,
        foundElems: number,
        totalElems: number,
        literalSearch: boolean,
        hasReturnArrow: boolean,
        correction: string|null,
        proposeCorrectionFrom: string|null,
        proposeCorrectionTo: string|null,
        typeFingerprint: Uint32Array,
        error: Array<string> | null,
    }

    /**
     * An entry in the search index database.
     */
    interface Row {
        crate: string,
        descShard: SearchDescShard,
        id: number,
        name: string,
        normalizedName: string,
        word: string,
        paramNames: string[],
        parent: ({ty: number, name: string, path: string, exactPath: string}|null|undefined),
        path: string,
        ty: number,
        type: FunctionSearchType | null,
    }

    /**
     * The viewmodel for the search engine results page.
     */
    interface ResultsTable {
        in_args: Array<ResultObject>,
        returned: Array<ResultObject>,
        others: Array<ResultObject>,
        query: ParsedQuery,
    }

    type Results = Map<String, ResultObject>;

    /**
     * An annotated `Row`, used in the viewmodel.
     */
    interface ResultObject {
        desc: string,
        displayPath: string,
        fullPath: string,
        href: string,
        id: number,
        dist: number,
        path_dist: number,
        name: string,
        normalizedName: string,
        word: string,
        index: number,
        parent: (Object|undefined),
        path: string,
        ty: number,
        type?: FunctionSearchType,
        paramNames?: string[],
        displayTypeSignature: Promise<rustdoc.DisplayTypeSignature> | null,
        item: Row,
        dontValidate?: boolean,
    }

    /**
     * output of `formatDisplayTypeSignature`
     */
    interface DisplayTypeSignature {
        type: Array<string>,
        mappedNames: Map<string, string>,
        whereClause: Map<string, Array<string>>,
    }

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
     */
    type RawFunctionSearchType =
        0 |
        [(number|Array<RawFunctionType>)] |
        [(number|Array<RawFunctionType>), (number|Array<RawFunctionType>)] |
        Array<(number|Array<RawFunctionType>)>
    ;

    /**
     * A single function input or output type. This is either a single path ID, or a pair of
     * [path ID, generics].
     *
     * Numeric IDs are *ONE-indexed* into the paths array (`p`). Zero is used as a sentinel for `null`
     * because `null` is four bytes while `0` is one byte.
     */
    type RawFunctionType = number | [number, Array<RawFunctionType>];

    /**
     * The type signature entry in the decoded search index.
     * (The "Raw" objects are encoded differently to save space in the JSON).
     */
    interface FunctionSearchType {
        inputs: Array<FunctionType>,
        output: Array<FunctionType>,
        where_clause: Array<Array<FunctionType>>,
    }

    /**
     * A decoded function type, made from real objects.
     * `ty` will be negative for generics, positive for types, and 0 for placeholders.
     */
    interface FunctionType {
        id: null|number,
        ty: number|null,
        name?: string,
        path: string|null,
        exactPath: string|null,
        unboxFlag: boolean,
        generics: Array<FunctionType>,
        bindings: Map<number, Array<FunctionType>>,
    };

    interface HighlightedFunctionType extends FunctionType {
        generics: HighlightedFunctionType[],
        bindings: Map<number, HighlightedFunctionType[]>,
        highlighted?: boolean;
    }

    interface FingerprintableType {
        id: number|null;
        generics: FingerprintableType[];
        bindings: Map<number, FingerprintableType[]>;
    };

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
     * `f` contains function signatures, or `0` if the item isn't a function.
     * More information on how they're encoded can be found in rustc-dev-guide
     *
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
     * The first item is the type, the second is the name, the third is the visible path (if any) and
     * the fourth is the canonical path used for deduplication (if any).
     *
     * `r` is the canonical path used for deduplication of re-exported items.
     * It is not used for associated items like methods (that's the fourth element
     * of `p`) but is used for modules items like free functions.
     *
     * `c` is an array of item indices that are deprecated.
     */
    type RawSearchIndexCrate = {
    doc: string,
    a: { [key: string]: number[] },
    n: Array<string>,
    t: string,
    D: string,
    e: string,
    q: Array<[number, string]>,
    i: string,
    f: string,
    p: Array<[number, string] | [number, string, number] | [number, string, number, number] | [number, string, number, number, string]>,
    b: Array<[number, String]>,
    c: string,
    r: Array<[number, number]>,
    P: Array<[number, string]>,
    };

    type VlqData = VlqData[] | number;

    /**
     * Maps from crate names to trait implementation data.
     * Provied by generated `trait.impl` files.
     */
    type Implementors = {
        [key: string]: Array<[string, number, Array<string>]>
    }

    type TypeImpls = {
        [cratename: string]: Array<Array<string|0>>
    }

    /**
     * Directory structure for source code view,
     * defined in generated `src-files.js`.
     *
     * is a tuple of (filename, subdirs, filenames).
     */
    type Dir = [string, rustdoc.Dir[], string[]]

    /**
     * Indivitual setting object, used in `settings.js`
     */
    interface Setting {
        js_name: string,
        name: string,
        options?: string[],
        default: string | boolean,
    }
}
