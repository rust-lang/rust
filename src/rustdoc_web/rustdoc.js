/*jslint node: true, stupid: true, es5: true, regexp: true, nomen: true */
"use strict";

/*
 * This file is part of the rustdoc_web package.
 *
 * (c) Jordi Boggiano <j.boggiano@seld.be>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

var Twig = require("twig"),
    fs = require("fs"),
    globsync = require('glob-whatev');

// params (inputDir, outputDir, faviconUrl, logoUrl, menu, title, baseSourceUrls, knownCrates)
var config = JSON.parse(fs.readFileSync('config.json'));
config.inputDir = config.inputDir || "input";
config.outputDir = config.outputDir.replace(/\/*$/, '/') || "build/";
config.logoUrl = config.logoUrl || "http://www.rust-lang.org/logos/rust-logo-128x128-blk.png";
config.baseSourceUrls = config.baseSourceUrls || {};
config.menu = config.menu || [];
config.title = config.title || '';
config.rustVersion = config.rustVersion || '0.8'; // TODO should be current once the latest version is built as current
config.knownCrates = config.knownCrates || {};
config.docBlockRenderer = 'markdown';

var renderDocs;

if (config.docBlockRenderer === 'markdown') {
    renderDocs = (function () {
        var marked = require('marked'),
            hljs = require('highlight.js'),
            options = {
                gfm: true,
                highlight: function (code, lang) {
                    return hljs.highlightAuto(code).value;
                    //return hljs.highlight(lang || 'rust', code).value;
                },
                tables: true,
                breaks: false,
                pedantic: false,
                sanitize: false,
                smartLists: true,
                smartypants: false,
                langPrefix: 'lang-'
            };

        marked.setOptions(options);

        return function (str) {
            var tokens;
            if (str === '') {
                return str;
            }
            tokens = marked.lexer(str, options);
            return marked.parser(tokens);
        };
    }());
} else {
    throw new Error('Invalid docblock renderer: ' + config.docBlockRenderer);
}

// merge in default known crates
[
    {name: 'std', url: "http://seld.be/rustdoc/%rustVersion%/", type: "rustdoc_web"},
    {name: 'extra', url: "http://seld.be/rustdoc/%rustVersion%/", type: "rustdoc_web"},
].forEach(function (crate) {
    if (config.knownCrates[crate.name] === undefined) {
        config.knownCrates[crate.name] = {url: crate.url, type: crate.type};
    }
});

var transTexts = {
    'mods': 'Modules',
    'structs': 'Structs',
    'enums': 'Enums',
    'traits': 'Traits',
    'typedefs': 'Type Definitions',
    'statics': 'Statics',
    'fns': 'Functions',
    'reexports': 'Re-exports',
    'crates': 'Crates',
    'mod': 'Module',
    'struct': 'Struct',
    'enum': 'Enum',
    'trait': 'Trait',
    'typedef': 'Type Definition',
    'static': 'Static',
    'fn': 'Function',
    'reexport': 'Re-export',
};

Twig.extendFilter('trans', function (str) {
    if (transTexts[str] !== undefined) {
        return transTexts[str];
    }

    return str;
});

Twig.extendFilter('substring', function (str, args) {
    var from = args[0], to = args[1];
    if (to < 0) {
        to = str.length + to;
    }

    return str.substring(from, to);
});

function createTypeTreeNode(name, parent) {
    parent = parent || null;

    return {
        // special metadata
        name: name,
        parent: parent,
        submods: {},

        // list of elements
        mods: {},
        structs: {},
        enums: {},
        traits: {},
        typedefs: {},
        fns: {},
        reexports: {},
        statics: {},

        // reuse the instance of the parent tree so all trees share the impls because they need to be discoverable across the board
        impls: parent ? parent.impls : {}, // implementations for a given struct
        implsfor: parent ? parent.implsfor : {}, // implementations of traits for a given struct
        implsof: parent ? parent.implsof : {}, // implementations of a given trait
    };
}

function shortenType(type) {
    return type.match(/s$/) ? type.substring(0, type.length - 1) : type;
}

function extract(data, key) {
    var res = '';
    data.forEach(function (attr) {
        if ((attr.variant === "NameValue" || attr.variant === "List") && attr.fields[0] === key) {
            res = attr.fields[1];
        }
    });

    return res;
}

function extractDocs(elem, skipFormatting, returnFalseIfNotDocable) {
    var docs = extract(elem.attrs, 'doc');

    if (docs instanceof Array && docs[0].fields[0] === 'hidden') {
        return returnFalseIfNotDocable === true ? false : '';
    }
    if (returnFalseIfNotDocable === true && elem.visibility === 'hidden') {
        return false;
    }

    docs = docs.toString();

    return skipFormatting === true ? docs : renderDocs(docs);
}

function shortDescription(elem, skipFormatting) {
    var match, docblock = extractDocs(elem, true);

    if (docblock === false) {
        return '';
    }

    match = docblock.match(/^([\s\S]+?)\r?\n[ \t\*]*\r?\n([\s\S]+)/);
    if (match) {
        docblock = match[1];
    }

    return skipFormatting === true ? docblock : renderDocs(docblock);
}

function getPath(tree) {
    var bits = [];
    bits.push(tree.name || '');
    while (tree.parent) {
        tree = tree.parent;
        bits.push(tree.name);
    }

    bits.reverse();

    return bits;
}

function getDecl(element) {
    if (element.decl !== undefined) {
        return element.decl;
    }

    return element.inner.fields[0].decl;
}

function getGenerics(element) {
    if (element.inner === undefined || element.inner.fields === undefined) {
        throw new Error('Invalid element: ' + JSON.stringify(element));
    }
    return element.inner.fields[0].generics;
}

function extractId(inner, element) {
    var ids = [];

    switch (inner) {
    case 'Unit':
    case 'Bool':
        return 'std::' + inner.toLowerCase();
    case 'String':
        return 'std::str';
    }

    switch (inner.variant) {
    case 'ResolvedPath':
        return inner.fields[2];
    case 'BorrowedRef':
        return extractId(inner.fields[2], element);
    case 'Unique':
        return extractId(inner.fields[0], element);
    case 'Vector':
        return 'std::vec';
    case 'RawPointer':
        return extractId(inner.fields[1], element);
    case 'Managed':
        return extractId(inner.fields[1], element);
    case 'Primitive':
        if (typeof inner.fields[0] === 'string') {
            return 'std::' + inner.fields[0].replace('ty_', '');
        }
        return 'std::' + inner.fields[0].variant.replace('ty_', '');
    case 'Tuple':
        return inner.fields[0].map(function (item) {
            return extractId(item, element);
        });
    case 'Generic':
        element.generics.type_params.forEach(function (param) {
            var bounds, id;
            if (param.id === inner.fields[0]) {
                bounds = param.bounds.filter(function (bound) {
                    return typeof bound !== 'string';
                });
                if (bounds.length === 0) {
                    return;
                }
                id = false;
                bounds.forEach(function (bound) {
                    if (bound.variant === 'TraitBound' && bound.fields[0].variant === 'ResolvedPath') {
                        id = bound.fields[0].fields[2];
                    } else if (bound.variant === 'TraitBound' && bound.fields[0].variant === 'External') {
                        id = bound.fields[0].fields[1] + ' ' + bound.fields[0].fields[0];
                    }
                });
                if (!id) {
                    throw new Error('Impl is for a Generic variant but the id could not be resolved in: ' + JSON.stringify(inner) + ' ' + JSON.stringify(element));
                }
                ids.push(id);
            }
        });

        return ids;
    case 'External':
        return inner.fields[1] + ' ' + inner.fields[0];
    default:
        throw new Error('Unknown variant: ' + inner.variant + ' in ' + JSON.stringify(inner));
    }
}

function primitiveType(type) {
    var foundType = typeof type === 'string' ? type.substring(3) : type.fields[0].substring(3),
        typeAliases = {
            u: 'uint',
            f: 'float',
            i: 'int',
        },
        knownTypes = [
            "char",
            "u", "u8", "u16", "u32", "u64",
            "i", "i8", "i16", "i32", "i64",
            "f", "f8", "f16", "f32", "f64"
        ];

    if (knownTypes.indexOf(foundType) === -1) {
        throw new Error('Unknown type: ' + JSON.stringify(type));
    }

    if (typeAliases[foundType] !== undefined) {
        return typeAliases[foundType];
    }

    return foundType;
}

function render(template, vars, references, version, cb) {
    function relativePath(fromTree, toTree) {
        var fromPath, toPath;

        if (fromTree === toTree) {
            return '';
        }

        fromPath = getPath(fromTree);
        toPath = getPath(toTree);

        while (toPath.length && fromPath.length && toPath[0] === fromPath[0]) {
            toPath.shift();
            fromPath.shift();
        }

        return (new Array(fromPath.length + 1).join('../') + toPath.join('/') + '/').replace(/\/+$/, '/');
    }

    function modPath(typeTree) {
        var path = getPath(typeTree).join('::');

        return path + (path ? '::' : '');
    }

    // helpers
    vars.short_description = shortDescription;
    vars.long_description = function (elem) {
        var match, docblock = extractDocs(elem, true);

        if (docblock === false) {
            return '';
        }

        match = docblock.match(/^([\s\S]+?)\r?\n[ \t\*]*\r?\n([\s\S]+)/);
        return match ? renderDocs(match[2]) : '';
    };
    vars.link_to_element = function (id, currentTree) {
        var modPrefix = '';
        if (!currentTree) {
            throw new Error('Missing currentTree arg #2');
        }
        modPrefix = modPath(references[id].tree);

        return '<a class="' + shortenType(references[id].type) + '" href="' + vars.url_to_element(id, currentTree) + '" title="' + modPrefix + references[id].def.name + '">' + references[id].def.name + '</a>';
    };
    vars.link_to_external = function (name, type, knownCrates, version) {
        var crate, path, match, url, localCrate;
        match = name.match(/^([^:]+)(::.*)?$/);
        crate = match[1];
        path = name.replace(/::/g, '/') + '.html';
        path = path.replace(/([^\/]+)$/, type + '.$1');

        version.crates.forEach(function (cr) {
            if (cr.name === crate) {
                localCrate = true;
            }
        });

        if (localCrate) { // crate is part of this build
            url = vars.root_path + version.version + '/' + path;
        } else { // crate is known at another URL
            if (knownCrates[crate] === undefined) {
                return name;
            }
            if (knownCrates[crate].type !== 'rustdoc_web') {
                console.log('WARNING: Unknown crate type ' + knownCrates[crate].type);
                return name;
            }

            url = knownCrates[crate].url
                .replace(/%rustVersion%/g, config.rustVersion)
                .replace(/%version%/g, version.version)
                .replace(/\/*$/, '/');
            url += path;
        }

        return '<a class="' + shortenType(type) + '" href="' + url + '">' + name + '</a>';
    };
    vars.element_by_id = function (id) {
        return references[id];
    };
    vars.url_to_element = function (id, currentTree) {
        if (!currentTree) {
            throw new Error('Missing currentTree arg #2');
        }
        if (references[id].type === 'mods') {
            return relativePath(currentTree, references[id].tree) + references[id].def.name + '/index.html';
        }
        return relativePath(currentTree, references[id].tree) + shortenType(references[id].type) + '.' + references[id].def.name + '.html';
    };
    vars.breadcrumb = function (typeTree, currentTree) {
        var path = [], out = '', tmpTree;

        currentTree = currentTree || typeTree;
        path.push(typeTree);

        tmpTree = typeTree;
        while (tmpTree.parent) {
            tmpTree = tmpTree.parent;
            path.push(tmpTree);
        }
        path.reverse();

        path.forEach(function (targetTree) {
            out += '&#8203;' + (out ? '::' : '') + '<a href="' + relativePath(currentTree, targetTree) + 'index.html">' + targetTree.name + '</a>';
        });

        return out + '::';
    };
    vars.filter_priv_traits = function (traits) {
        var pubTraits = [];
        if (traits === null || traits === undefined) {
            return traits;
        }

        traits.forEach(function (trait) {
            if (trait.visibility !== 'private') {
                pubTraits.push(trait);
            }
        });

        return pubTraits;
    };
    vars.filter_docable = function (elems, type) {
        var key, filtered = {};
        for (key in elems) {
            if (type === 'reexports') {
                if (elems[key].visibility === 'public') {
                    filtered[key] = elems[key];
                }
            } else if (extractDocs(references[key].def, true, true) !== false) {
                filtered[key] = elems[key];
            }
        }
        return filtered;
    };
    vars.extract_docs = extractDocs;
    vars.short_enum_type = function (type, currentTree) {
        if (type === 'CLikeVariant') {
            return '';
        }

        return vars.short_type(type, currentTree);
    };
    vars.sort = function (obj) {
        var key, elems = [];
        for (key in obj) {
            elems.push({id: key, name: obj[key]});
        }

        return elems.sort(function (a, b) {
            return a.name.localeCompare(b.name);
        });
    };
    vars.extract_parent_docs = function (impl, methodName) {
        var foundDocs = '';
        if (impl.inner.fields[0].trait_ === null) {
            return '';
        }
        if (impl.inner.fields[0].trait_.variant !== 'ResolvedPath') {
            return '';
        }

        references[impl.inner.fields[0].trait_.fields[2]].def.inner.fields[0].methods.forEach(function (method) {
            if (method.fields[0].name === methodName) {
                foundDocs = extractDocs(method.fields[0]);
            }
        });

        return foundDocs;
    };
    vars.unique_sorted_trait_impls = function (id, currentTree) {
        var impls, knownStructIds = [], uniqueImpls = [];

        impls = currentTree.implsof[id] || [];

        // TODO possibly collect implsof in a version.implsof instead of the tree,
        // so that crates can share it and we can display every implementor here

        impls.forEach(function (impl) {
            var structId;
            if (impl.inner.fields[0].for_.variant === 'ResolvedPath') {
                structId = impl.inner.fields[0].for_.fields[2];
                if (knownStructIds.indexOf(structId) === -1) {
                    uniqueImpls.push(impl);
                    knownStructIds.push(structId);
                }
            } else {
                uniqueImpls.push(impl);
            }
        });

        return uniqueImpls.sort(function (a, b) {
            var aResolved = a.inner.fields[0].for_.variant === 'ResolvedPath',
                bResolved = b.inner.fields[0].for_.variant === 'ResolvedPath';
            if (aResolved && bResolved) {
                return references[a.inner.fields[0].for_.fields[2]].def.name.localeCompare(references[b.inner.fields[0].for_.fields[2]].def.name);
            }

            if (aResolved) {
                return -1;
            }
            if (bResolved) {
                return 1;
            }

            return 0;
        });
    };
    vars.extract_required_methods = function (trait) {
        var res = [];
        trait.inner.fields[0].methods.forEach(function (method) {
            if (method.variant === "Required") {
                res.push(method.fields[0]);
            }
        });
        return res;
    };
    vars.extract_provided_methods = function (trait) {
        var res = [];
        trait.inner.fields[0].methods.forEach(function (method) {
            if (method.variant === "Provided") {
                res.push(method.fields[0]);
            }
        });
        return res;
    };
    vars.count = function (data) {
        var key, count = 0;
        if (data instanceof Array) {
            return data.length;
        }

        for (key in data) {
            count += 1;
        }

        return count;
    };
    vars.short_type = function shortType(type, currentTree, realType, pureLink) {
        var types, path;

        if (!currentTree) {
            throw new Error('Missing currentTree arg #2');
        }

        if (typeof type === 'string') {
            type = {variant: type, fields: []};
        }

        switch (realType || type.variant) {
        case 'Primitive':
            return primitiveType(type.fields[0]);

        case 'ResolvedPath':
            if (!references[type.fields[2]]) {
                throw new Error('Invalid resolved ref id: ' + type.fields[2]);
            }
            return vars.link_to_element(type.fields[2], currentTree) + (pureLink === true ? '' : vars.render_generics(type.fields[0], currentTree, 'arg'));

        case 'External':
            //                           external path   external type
            return vars.link_to_external(type.fields[0], type.fields[1], config.knownCrates, version);

        case 'Tuple':
        case 'TupleVariant':
            types = type.fields[0].map(function (t) {
                return shortType(t, currentTree, realType);
            }).join(', ');
            return '(' + types + ')';

        case 'ViewItemItem':
            return 'pub use ' + shortType(type.fields[0].inner, currentTree, realType) + ';';
        case 'Import':
            return type.fields[0].map(function (t) {
                return shortType(t, currentTree, realType);
            }).join(', ');
        case 'SimpleImport':
            path = type.fields[1].segments.map(function (s) {
                return s.name;
            }).join('::');
            if (type.fields[0] === path || path.substring(path.length - type.fields[0].length - 2) === '::' + type.fields[0]) {
                return path;
            }

            return type.fields[0] + ' = ' + path;
        case 'GlobImport':
            return type.fields[0].segments.map(function (s) { return s.name; }).join('::') + '::*';
        case 'ImportList':
            return type.fields[0].segments.map(function (s) { return s.name; }).join('::') + '::{' + type.fields[1].join(', ') + '}';
        case 'String':
            return 'str';
        case 'Bool':
            return 'bool';
        case 'Managed':
            return '@' + (type.fields[0] === 'Mutable' ? 'mut ' : '') + shortType(type.fields[1], currentTree, realType);
        case 'RawPointer':
            return '*' + (type.fields[0] === 'Mutable' ? 'mut ' : '') + shortType(type.fields[1], currentTree, realType);
        case 'BorrowedRef':
            return '&amp;' + (type.fields[0] ? "'" + type.fields[0]._field0 + ' ' : '') + (type.fields[1] === 'Mutable' ? 'mut ' : '') + shortType(type.fields[2], currentTree, realType);
        case 'Unique':
            return '~' + shortType(type.fields[0], currentTree, realType);
        case 'Vector':
            return '[' + shortType(type.fields[0], currentTree, realType) + ']';
        case 'FixedVector':
            return '[' + shortType(type.fields[0], currentTree, realType) + ', ..' + type.fields[1] + ']';
        case 'Bottom':
            return '!';
        case 'Self':
            return 'Self';
        case 'SelfStatic':
            return '';
        case 'SelfValue':
            return 'self';
        case 'SelfOwned':
            return '~self';
        case 'SelfManaged':
            return '@' + (type.fields[0] === 'Mutable' ? 'mut ' : '') + 'self';
        case 'SelfBorrowed':
            return '&amp;' + (type.fields[0] ? type.fields[0]._field0 + ' ' : '') + (type.fields[1] === 'Mutable' ? 'mut ' : '') + 'self';
        case 'Closure':
            return vars.render_fn(type.fields[0], currentTree, 'Closure');
        case 'Generic':
            if (references[type.fields[0]] === undefined) {
                throw new Error('Invalid generic reference id in ' + JSON.stringify(type));
            }
            return references[type.fields[0]].def.name;
        case 'Unit':
            return '()';
        case 'BareFunction':
            return (type.fields[0].abi ? 'extern ' + type.fields[0].abi + ' ' : '') + vars.render_fn(type.fields[0], currentTree, 'BareFunction');
        }

        throw new Error('Can not render short type: ' + (realType || type.variant) + ' ' + JSON.stringify(type));
    };
    vars.render_fn = function (fn, currentTree, fnType) {
        var output = '', decl = getDecl(fn), temp;

        if (fn.inner && fn.inner.fields && fn.inner.fields[0].purity === 'unsafe_fn') {
            output += 'unsafe ';
        }

        if (fnType === 'Closure') {
            if (fn.sigil) {
                if (fn.sigil === 'BorrowedSigil') {
                    output += '&amp;';
                } else if (fn.sigil === 'ManagedSigil') {
                    output += '@';
                } else if (fn.sigil === 'OwnedSigil') {
                    output += '~';
                } else {
                    throw new Error('Unknown sigil type ' + fn.sigil);
                }
            }
            if (fn.region) {
                output += "'" + fn.region._field0 + " ";
            }
            if (fn.onceness === 'once') {
                output += 'once ';
            }
        }

        output += 'fn' + (fn.name ? ' <strong class="fnname">' + fn.name + '</strong>' : '') + vars.render_generics(fn, currentTree, fnType) + '(\n    ';
        if (fn.inner && fn.inner.fields && fn.inner.fields[0].self_) {
            temp = vars.short_type(fn.inner.fields[0].self_, currentTree);
            if (temp) {
                output += temp;
                if (decl.inputs.length > 0) {
                    output += ', \n    ';
                }
            }
        }
        output += decl.inputs.map(function (arg) {
            return (arg.name ? arg.name + ': ' : '') + vars.short_type(arg.type_, currentTree);
        }).join(', \n    ');
        output += '\n)';

        if (decl.output !== 'Unit') {
            output += ' -&gt; ' + vars.short_type(decl.output, currentTree);
        }

        Twig.extend(function (Twig) {
            if (Twig.lib.strip_tags(output).replace(/&(gt|lt)/g, '').length < 100 || fnType !== 'fn') {
                output = output.replace(/\n {4}|\n/g, '');
            }
        });

        return output;
    };
    vars.collect_impls = function (element, currentTree) {
        return {
            impls: currentTree.impls[element.id],
            trait_impls: vars.filter_priv_traits(currentTree.implsfor[element.id])
        };
    };
    vars.render_generics = function renderGenerics(element, currentTree, type) {
        var type_params, lifetimes, output = '', generics;

        if (type === 'Closure') {
            generics = {type_params: [], lifetimes: element.lifetimes};
        } else if (type === 'BareFunction') {
            generics = element.generics;
        } else if (type === 'arg' || type === 'bound') {
            generics = {type_params: element.typarams, lifetimes: element.lifetime ? [element.lifetime] : null};
        } else {
            generics = getGenerics(element);
        }
        if (!generics) {
            throw new Error('Element has invalid generics defined ' + JSON.stringify(element));
        }

        function renderBounds(bound) {
            var res = '';
            if (bound === 'RegionBound') {
                return "'static";
            }
            if (bound.variant === 'TraitBound') {
                return vars.short_type(bound.fields[0], currentTree);
            }

            if (bound.fields === undefined || bound.fields[0].path === undefined) {
                throw new Error("Unknown bound type " + JSON.stringify(bound));
            }
            bound = bound.fields[0].path;

            if (bound.name) {
                res += bound.name;
            }
            res += renderGenerics(bound, currentTree, 'bound');

            return res;
        }

        type_params = generics.type_params || [];
        lifetimes = generics.lifetimes || [];

        if (type_params.length || lifetimes.length) {
            output += '&lt;';
            if (lifetimes.length) {
                output += "'" + lifetimes.map(function (l) { return l._field0; }).join(", '");
            }
            if (type_params.length && lifetimes.length) {
                output += ', ';
            }
            output += type_params.map(function (t) {
                var res;
                if (t.name) {
                    res = t.name;
                } else {
                    res = vars.short_type(t, currentTree);
                }
                if (t.bounds && t.bounds[0] !== undefined) {
                    res += ": " + t.bounds.map(renderBounds).join(' + ');
                }

                return res;
            }).join(', ');
            output += '&gt;';
        }

        return output;
    };
    vars.source_url = function (element, crate) {
        var matches;
        if (!element.source) {
            throw new Error('Element has no source: ' + JSON.stringify(element));
        }
        if (element.source.match(/^<std-macros>:/)) {
            return '';
        }
        matches = element.source.match(/^([a-z0-9_.\/\-]+):(\d+):\d+:? (\d+):\d+$/i);
        if (!matches) {
            throw new Error('Could not parse element.source for ' + JSON.stringify(element));
        }

        return config.baseSourceUrls[crate.name].replace('%version%', crate.version) + matches[1] + '#L' + matches[2] + '-' + matches[3];
    };

    cb(
        Twig.twig({
            path: "templates/" + template,
            async: false,
            base: "templates/",
            strict_variables: true
        }).render(vars)
    );
}

function indexModule(path, module, typeTree, references, searchIndex) {
    var uid = 1,
        delayedIndexations = [],
        types = {
            ModuleItem: 'mods',
            StructItem: 'structs',
            EnumItem: 'enums',
            TraitItem: 'traits',
            TypedefItem: 'typedefs',
            FunctionItem: 'fns',
            StaticItem: 'statics',
            ImplItem: 'impls',
            ViewItemItem: 'reexports',
        };

    function indexTyparams(typarams) {
        typarams.forEach(function (typaram) {
            references[typaram.id] = {type: 'typaram', def: typaram, tree: typeTree};
        });
    }

    function indexMethods(methods, parentName, parentType) {
        methods.forEach(function (method) {
            var generics;

            method = method.fields ? method.fields[0] : method;
            generics = getGenerics(method);
            if (generics && generics.type_params) {
                indexTyparams(generics.type_params);
            }
            searchIndex.push({type: 'method', name: method.name, parent: parentName, parentType: parentType, desc: shortDescription(method, true), path: getPath(typeTree).join('::')});
        });
    }

    function indexVariants(variants, parentName, parentType) {
        variants.forEach(function (variant) {
            searchIndex.push({type: 'variant', name: variant.name, parent: parentName, parentType: parentType, desc: '', path: getPath(typeTree).join('::')});
        });
    }

    function indexImpl(def) {
        var generics, forId = null, ofId = null;

        // TODO fix this and find a way to list fn impls on the module page maybe?
        if (['Closure', 'BareFunction'].indexOf(def.inner.fields[0].for_.variant) !== -1) {
            return;
        }

        forId = extractId(def.inner.fields[0].for_, def.inner.fields[0]);

        if (def.inner.fields[0].trait_) {
            ofId = extractId(def.inner.fields[0].trait_, def.inner.fields[0]);
            if (!ofId) {
                throw new Error('Unknown trait definition: ' + JSON.stringify(def));
            }
        }

        generics = getGenerics(def);
        if (generics && generics.type_params) {
            indexTyparams(generics.type_params);
        }

        if (!forId) {
            throw new Error('An impl must be for a struct|enum|fn|trait, no forId found in: ' + JSON.stringify(def));
        }

        forId = forId instanceof Array ? forId : [forId];

        // TODO support entries that are for nothing (e.g. impl<T> Clone for *T) and list them on the mod page
        if (forId.length === 0) {
            return;
        }

        if (ofId) {
            ofId = ofId instanceof Array ? ofId : [ofId];

            forId.forEach(function (id) {
                if (typeTree.implsfor[id] === undefined) {
                    typeTree.implsfor[id] = [];
                }
                typeTree.implsfor[id].push(def);
            });
            ofId.forEach(function (id) {
                if (typeTree.implsof[id] === undefined) {
                    typeTree.implsof[id] = [];
                }
                typeTree.implsof[id].push(def);
            });
        } else {
            forId.forEach(function (id) {
                if (typeTree.impls[id] === undefined) {
                    typeTree.impls[id] = [];
                }
                typeTree.impls[id].push(def);
            });
        }

        forId.forEach(function (id) {
            delayedIndexations.push(function () {
                if (references[id] === undefined) {
                    indexMethods(def.inner.fields[0].methods, id, 'primitive');
                } else {
                    indexMethods(def.inner.fields[0].methods, references[id].def.name, shortenType(references[id].type));
                }
            });
        });
    }

    function indexItem(type, def) {
        var name = def.name, generics;

        if (type === 'impls') {
            indexImpl(def);
            return;
        }

        if (type === 'reexports') {
            typeTree[type][uid] = def;
            uid += 1;
            return;
        }
        if (type === 'mods') {
            // this is a meaningful id: std::vec and other primitives can find their impls
            // since extractId returns the path of the module for primitives
            def.id = path + '::' + name;
            typeTree.submods[name] = createTypeTreeNode(name, typeTree);
            delayedIndexations = delayedIndexations.concat(indexModule(path + '::' + name, def, typeTree.submods[name], references, searchIndex));
        } else if (type === 'statics') {
            def.id = path + '::' + name;
        } else if (def.id === undefined) {
            throw new Error('Missing id on type ' + type + ' content: ' + JSON.stringify(def));
        }

        generics = getGenerics(def);
        if (generics && generics.type_params) {
            indexTyparams(generics.type_params);
        }
        if (type === 'traits') {
            indexMethods(def.inner.fields[0].methods, name, shortenType(type));
        }
        if (type === 'enums') {
            indexVariants(def.inner.fields[0].variants, name, shortenType(type));
        }

        typeTree[type][def.id] = name;
        searchIndex.push({type: shortenType(type), name: name, path: getPath(typeTree).join('::'), desc: shortDescription(def, true)});
        references[def.id] = {type: type, def: def, tree: typeTree};
    }

    if (module.inner.variant !== 'ModuleItem') {
        throw new Error('Invalid module, should contain an inner module item');
    }

    module.inner.fields.forEach(function (field) {
        field.items.forEach(function (item) {
            if (types[item.inner.variant] === undefined) {
                throw new Error('Unknown variant ' + item.inner.variant);
            }
            indexItem(types[item.inner.variant], item);
        });
    });

    return delayedIndexations;
}

function dumpModule(path, module, typeTree, references, crate, crates, version, versions) {
    var rootPath, matches, types,
        buildPath = config.outputDir + crate.version + "/" + path.replace(/::/g, '/') + '/';

    types = {
        ModuleItem: 'mods',
        StructItem: 'structs',
        EnumItem: 'enums',
        TraitItem: 'traits',
        TypedefItem: 'typedefs',
        FunctionItem: 'fns',
        StaticItem: false,
        ImplItem: false,
        ViewItemItem: false,
    };

    matches = path.match(/::/g);
    rootPath = '../../' + (matches ? new Array(matches.length + 1).join('../') : '');

    function renderTemplate(type, def, filename) {
        var data, cb;
        data = {
            path: path,
            type_tree: typeTree,
            type: shortenType(type),
            root_path: rootPath,
            element: def,
            crates: crates,
            crate: crate,
            versions: versions,
            cur_version: crate.version,
            config: config,
        };
        if (!fs.existsSync(buildPath)) {
            fs.mkdirSync(buildPath);
        }
        cb = function (out) {
            fs.writeFileSync(buildPath + filename, out);
        };
        render(type + '.twig', data, references, version, cb);
    }

    renderTemplate('mods', module, "index.html");

    module.inner.fields.forEach(function (field) {
        field.items.forEach(function (item) {
            var type = types[item.inner.variant];
            if (type === undefined) {
                throw new Error('Unknown variant ' + item.inner.variant);
            }
            if (type === false) {
                return;
            }

            if (type === 'mods') {
                dumpModule(path + '::' + item.name, item, typeTree.submods[item.name], references, crate, crates, version, versions);
            } else {
                renderTemplate(type, item, shortenType(type) + '.' + item.name + '.html');
            }
        });
    });
}

function renderCratesIndex(version, versions) {
    var data, cb;
    data = {
        root_path: '../',
        crates: version.crates,
        config: config,
        versions: versions,
        cur_version: version.version,
        // dummy object because we are not in a crate but the layout needs one
        crate: {version: version.version}
    };
    cb = function (out) {
        fs.writeFile(config.outputDir + version.version + '/index.html', out);
    };

    if (version.crates.length === 1) {
        cb('<DOCTYPE html><html><head><meta http-equiv="refresh" content="0; url=' + version.crates[0].name + '/index.html"></head><body></body></html>');
    } else {
        render('crates.twig', data, {}, version, cb);
    }
}

function renderVersionsIndex(versions) {
    var data, cb;
    data = {
        root_path: '',
        versions: versions,
        config: config,
    };
    cb = function (out) {
        fs.writeFile(config.outputDir + '/index.html', out);
    };
    if (versions.length === 1) {
        cb('<DOCTYPE html><html><head><meta http-equiv="refresh" content="0; url=' + versions[0].version + '/index.html"></head><body></body></html>');
    } else {
        render('versions.twig', data, {}, {}, cb);
    }
}

function initCrate(crate, searchIndex) {
    var sourceUrl, delayedIndexations, data = JSON.parse(fs.readFileSync(crate.path));
    if (data.schema !== '0.8.0') {
        throw new Error('Unsupported schema ' + data.schema);
    }

    crate.name = data.crate.name;
    crate.data = data.crate.module;
    crate.typeTree = createTypeTreeNode(crate.name);
    crate.references = {};
    crate.data.name = crate.name;
    crate.license = extract(data.crate.module.attrs, 'license').toString();

    // read the link.url of the crate and take that as default if the config has no url configured
    sourceUrl = extract(data.crate.module.attrs, 'link');
    if (sourceUrl !== '' && config.baseSourceUrls[crate.name] === undefined) {
        sourceUrl = extract(sourceUrl, 'url').toString();
        if (sourceUrl !== '') {
            config.baseSourceUrls[crate.name] = sourceUrl.replace(/\/*$/, '/');
        }
    }

    delayedIndexations = indexModule(crate.name, crate.data, crate.typeTree, crate.references, searchIndex);
    delayedIndexations.forEach(function (cb) {
        cb();
    });
}

function dumpCrate(crate, crates, version, versions) {
    dumpModule(crate.name, crate.data, crate.typeTree, crate.references, crate, crates, version, versions);
}

(function main() {
    var versions = [];

    if (!fs.existsSync(config.outputDir)) {
        fs.mkdirSync(config.outputDir);
    }

    globsync.glob(config.inputDir.replace(/\/*$/, '/') + '*').forEach(function (path) {
        var crates = [],
            version = path.replace(/.*?\/([^\/]+)\/$/, '$1');

        globsync.glob(path + '*.json').forEach(function (path) {
            var crate = path.replace(/.*?\/([^\/]+)\.json$/, '$1');
            crates.push({path: path, version: version});
        });

        versions.push({
            version: version,
            crates: crates,
            prerelease: null === require('semver').valid(version),
        });

        if (!fs.existsSync(config.outputDir + version)) {
            fs.mkdirSync(config.outputDir + version);
        }
    });

    versions.sort(function (a, b) {
        if (!a.prerelease && !b.prerelease) {
            return require('semver').rcompare(a.version, b.version);
        }

        if (a.prerelease && !b.prerelease) {
            return 1;
        }

        if (b.prerelease) {
            return -1;
        }

        return 0;
    });

    versions.forEach(function (version) {
        var searchIndex = [];

        version.crates.forEach(function (crate) {
            initCrate(crate, searchIndex);
        });

        version.crates.sort(function (a, b) {
            return a.name.localeCompare(b.name);
        });

        fs.writeFile(config.outputDir + version.version + '/search-index.js', "searchIndex = " + JSON.stringify(searchIndex));
        searchIndex = [];

        renderCratesIndex(version, versions);

        version.crates.forEach(function (crate) {
            console.log('Dumping ' + crate.name + ' for ' + version.version);
            dumpCrate(crate, version.crates, version, versions);
        });
    });
    renderVersionsIndex(versions);
}());
