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
    fns: fnlist
};

type fndoc = ~{
    id: ast_id,
    name: str,
    brief: option<str>,
    desc: option<str>,
    args: [argdoc],
    return: option<retdoc>
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

// Just to break the structural recursive types
enum modlist = [moddoc];
enum fnlist = [fndoc];
