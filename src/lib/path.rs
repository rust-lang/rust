
type path = str;

fn dirname(path p) -> path {
    auto sep = os.path_sep();
    check (_str.byte_len(sep) == 1u);
    let int i = _str.rindex(p, sep.(0));
    if (i == -1) {
        ret p;
    }
    ret _str.substr(p, 0u, i as uint);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
