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
    desc: option::t<str>,
    return: option::t<str>,
    args: map::hashmap<str, str>
};

// Just to break the structural recursive types
tag modlist = [moddoc];
tag fnlist = [fndoc];
