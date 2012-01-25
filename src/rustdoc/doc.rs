#[doc = "The document model"];

type ast_id = int;

type cratedoc = ~{
    topmod: moddoc,
};

type moddoc = ~{
    id: ast_id,
    name: str,
    path: [str],
    brief: option<str>,
    desc: option<str>,
    mods: modlist,
    fns: fnlist,
    consts: constlist,
    enums: enumlist
};

type constdoc = ~{
    id: ast_id,
    name: str,
    brief: option<str>,
    desc: option<str>,
    ty: option<str>
};

type fndoc = ~{
    id: ast_id,
    name: str,
    brief: option<str>,
    desc: option<str>,
    args: [argdoc],
    return: retdoc,
    failure: option<str>,
    sig: option<str>
};

type argdoc = ~{
    name: str,
    desc: option<str>,
    ty: option<str>
};

type retdoc = {
    desc: option<str>,
    ty: option<str>
};

type enumdoc = ~{
    id: ast_id,
    name: str,
    brief: option<str>,
    desc: option<str>,
    variants: [variantdoc]
};

type variantdoc = ~{
    name: str,
    desc: option<str>,
    sig: option<str>
};

// Just to break the structural recursive types
enum modlist = [moddoc];
enum constlist = [constdoc];
enum fnlist = [fndoc];
enum enumlist = [enumdoc];