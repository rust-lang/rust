type ast_id = int;

type cratedoc = ~{
    topmod: moddoc,
};

type moddoc = ~{
    name: str,
    mods: modlist,
    fns: fnlist
};

type fndoc = ~{
    id: ast_id,
    name: str,
    brief: str,
    desc: option<str>,
    return: option<retdoc>,
    args: [(str, str)]
};

type retdoc = {
    desc: option<str>,
    ty: option<str>
};

// Just to break the structural recursive types
tag modlist = [moddoc];
tag fnlist = [fndoc];
