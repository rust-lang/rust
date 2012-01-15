type cratedoc = {
    mods: [moddoc]
};

type moddoc = {
    fns: [fndoc]
};

type fndoc = {
    brief: str,
    desc: option::t<str>,
    return: option::t<str>,
    args: map::hashmap<str, str>
};
