import std::vec;
import std::option;
import front::ast;

export get_linkage_metas;
export find_attrs_by_name;

// From a list of crate attributes get only the meta_items that impact crate
// linkage
fn find_linkage_metas(vec[ast::attribute] attrs) -> vec[@ast::meta_item] {
    let vec[@ast::meta_item] metas = [];
    for (ast::attribute attr in find_attrs_by_name(attrs, "link")) {
        alt (attr.node.value.node) {
            case (ast::meta_list(_, ?items)) {
                metas += items;
            }
            case (_) {
                // FIXME: Maybe need a warning that this attr isn't
                // being used for linkage
            }
        }
    }
    ret metas;
}

// Search a list of attributes and return only those with a specific name
fn find_attrs_by_name(vec[ast::attribute] attrs,
                      ast::ident name) -> vec[ast::attribute] {
    auto filter = bind fn(&ast::attribute a,
                          ast::ident name) -> option::t[ast::attribute] {
        if (get_attr_name(a) == name) {
            option::some(a)
        } else {
            option::none
        }
    } (_, name);
    ret vec::filter_map(filter, attrs);
}

fn get_attr_name(&ast::attribute attr) -> ast::ident {
    get_meta_item_name(@attr.node.value)
}

fn find_meta_items_by_name(vec[@ast::meta_item] metas,
                           ast::ident name) -> vec[@ast::meta_item] {
    auto filter = bind fn(&@ast::meta_item m,
                          ast::ident name) -> option::t[@ast::meta_item] {
        if (get_meta_item_name(m) == name) {
            option::some(m)
        } else {
            option::none
        }
    } (_, name);
    ret vec::filter_map(filter, metas);
}

fn get_meta_item_name(&@ast::meta_item meta) -> ast::ident {
    alt (meta.node) {
        case (ast::meta_word(?n)) { n }
        case (ast::meta_name_value(?n, _)) { n }
        case (ast::meta_list(?n, _)) { n }
    }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
