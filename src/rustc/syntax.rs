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

import rustsyntax::print;
export print;

import rustsyntax::parse;
export parse;

export ast;
export ast_util;
export visit;
export fold;
export ext;
export util;

mod util {
    import rustsyntax::util::interner;
    export interner;
}