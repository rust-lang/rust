export = stringdex;

declare namespace stringdex {
    /**
     * The client interface to Stringdex.
     */
    interface Database {
        getData(colname: string): DataColumn|undefined;
    }
    /**
     * A compressed node in the search tree.
     *
     * This object logically addresses two interleaved trees:
     * a "prefix tree", and a "suffix tree". If you ask for
     * generic matches, you get both, but if you ask for one
     * that excludes suffix-only entries, you'll get prefixes
     * alone.
     */
    interface Trie {
        matches(): RoaringBitmap;
        substringMatches(): AsyncGenerator<RoaringBitmap>;
        prefixMatches(): AsyncGenerator<RoaringBitmap>;
        keysExcludeSuffixOnly(): Uint8Array;
        childrenExcludeSuffixOnly(): [number, Promise<Trie>][];
        child(id: number): Promise<Trie>?;
    }
    /**
     * The client interface to Stringdex.
     */
    interface DataColumn {
        isEmpty(id: number): boolean;
        at(id: number): Promise<Uint8Array|undefined>;
        search(name: Uint8Array|string): Promise<Trie?>;
        searchLev(name: Uint8Array|string): AsyncGenerator<Trie>;
        length: number,
    }
    /**
     * Callbacks for a host application and VFS backend.
     *
     * These functions are calleb with mostly-raw data,
     * except the JSONP wrapper is removed. For example,
     * a file with the contents `rr_('{"A":"B"}')` should,
     * after being pulled in, result in the `rr_` callback
     * being invoked.
     *
     * The success callbacks don't need to supply the name of
     * the file that succeeded, but, if you want successful error
     * reporting, you'll need to remember which files are
     * in flight and report the filename as the first parameter.
     */
    interface Callbacks {
        /**
         * Load the root of the search database
         * @param {string} dataString
         */
        rr_: function(string);
        err_rr_: function(any);
        /**
         * Load a nodefile in the search tree.
         * A node file may contain multiple nodes;
         * each node has five fields, separated by newlines.
         * @param {string} inputBase64
         */
        rn_: function(string);
        err_rn_: function(string, any);
        /**
         * Load a database column partition from a string
         * @param {string} dataString
         */
        rd_: function(string);
        err_rd_: function(string, any);
        /**
         * Load a database column partition from base64
         * @param {string} dataString
         */
        rb_: function(string);
        err_rb_: function(string, any);
    };
    /**
     * Hooks that a VFS layer must provide for stringdex to load data.
     *
     * When the root is loaded, the Callbacks object is provided. These
     * functions should result in callback functions being called with
     * the contents of the file, or in error callbacks being invoked with
     * the failed-to-load filename.
     */
    interface Hooks {
        /**
         * The first function invoked as part of loading a search database.
         * This function must, eventually, invoke `rr_` with the string
         * representation of the root file (the function call wrapper,
         * `rr_('` and `')`, must be removed).
         *
         * The supplied callbacks object is used to feed search data back
         * to the search engine core. You have to store it, so that
         * loadTreeByHash and loadDataByNameAndHash can use it.
         *
         * If this fails, either throw an exception, or call `err_rr_`
         * with the error object.
         */
        loadRoot: function(Callbacks);
        /**
         * Load a subtree file from the search index.
         * 
         * If this function succeeds, call `rn_` on the callbacks
         * object. If it fails, call `err_rn_(hashHex, error)`.
         * 
         * @param {string} hashHex
         */
        loadTreeByHash: function(string);
        /**
         * Load a column partition from the search database.
         *
         * If this function succeeds, call `rd_` or `rb_` on the callbacks
         * object. If it fails, call `err_rd_(hashHex, error)`. or `err_rb_`.
         * To determine which one, the wrapping function call in the js file
         * specifies it.
         *
         * @param {string} columnName
         * @param {string} hashHex
         */
        loadDataByNameAndHash: function(string, string);
    };
    class RoaringBitmap {
        constructor(array: Uint8Array|null, start?: number);
        static makeSingleton(number: number);
        static everything(): RoaringBitmap;
        static empty(): RoaringBitmap;
        isEmpty(): boolean;
        union(that: RoaringBitmap): RoaringBitmap;
        intersection(that: RoaringBitmap): RoaringBitmap;
        contains(number: number): boolean;
        entries(): Generator<number>;
        first(): number|null;
        consumed_len_bytes: number;
    };

    type Stringdex = {
        /**
         * Initialize Stringdex with VFS hooks.
         * Returns a database that you can use.
         */
        loadDatabase: function(Hooks): Promise<Database>,
    };

    const Stringdex: Stringdex;
    const RoaringBitmap: Class<stringdex.RoaringBitmap>;
}

declare global {
    interface Window {
        Stringdex: stringdex.Stringdex;
        RoaringBitmap: Class<stringdex.RoaringBitmap>;
        StringdexOnload: Array<function(stringdex.Stringdex): any>?;
    };
}