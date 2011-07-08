import std::map;

type crate_metadata = rec(str name, vec[u8] data);

type cstore = @rec(map::hashmap[int, crate_metadata] metas,
                   vec[str] used_crate_files,
                   vec[str] used_libraries,
                   vec[str] used_link_args);

fn mk_cstore() -> cstore {
    auto meta_cache = map::new_int_hash[crate_metadata]();
    ret @rec(metas = meta_cache,
             used_crate_files = [],
             used_libraries = [],
             used_link_args = []);
}

fn get_crate_data(&cstore cstore, int cnum) -> crate_metadata {
    ret cstore.metas.get(cnum);
}

fn set_crate_data(&cstore cstore, int cnum, &crate_metadata data) {
    cstore.metas.insert(cnum, data);
}

fn have_crate_data(&cstore cstore, int cnum) -> bool {
    ret cstore.metas.contains_key(cnum);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
