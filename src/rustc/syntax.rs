// Temporary wrapper modules for migrating syntax to its own crate
import rustsyntax::codemap;
export codemap;

import rustsyntax::ast;
export ast;

import rustsyntax::ast_util;
export ast_util;

import rustsyntax::visit;
export visit;

import rustsyntax::fold;
export fold;

export ast;
export ast_util;
export visit;
export fold;
export parse;
export ext;
export print;
export util;

mod util {
    import rustsyntax::util::interner;
    export interner;
}