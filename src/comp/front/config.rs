import std::ivec;
import std::option;
import syntax::ast;
import syntax::fold;
import attr;

export strip_unconfigured_items;

// Support conditional compilation by transforming the AST, stripping out
// any items that do not belong in the current configuration
fn strip_unconfigured_items(@ast::crate crate) -> @ast::crate {
    auto cfg = crate.node.config;

    auto precursor = rec(fold_mod = bind fold_mod(cfg, _, _),
                         fold_block = bind fold_block(cfg, _, _),
                         fold_native_mod = bind fold_native_mod(cfg, _, _)
                         with *fold::default_ast_fold());

    auto fold = fold::make_fold(precursor);
    auto res = @fold.fold_crate(*crate);
    // FIXME: This is necessary to break a circular reference
    fold::dummy_out(fold);
    ret res;
}

fn filter_item(&ast::crate_cfg cfg,
               &@ast::item item) -> option::t[@ast::item] {
    if (item_in_cfg(cfg, item)) {
        option::some(item)
    } else {
        option::none
    }
}

fn fold_mod(&ast::crate_cfg cfg, &ast::_mod m,
              fold::ast_fold fld) -> ast::_mod {
    auto filter = bind filter_item(cfg, _);
    auto filtered_items = ivec::filter_map(filter, m.items);
    ret rec(view_items=ivec::map(fld.fold_view_item, m.view_items),
            items=ivec::map(fld.fold_item, filtered_items));
}

fn filter_native_item(&ast::crate_cfg cfg, &@ast::native_item item)
    -> option::t[@ast::native_item] {
    if (native_item_in_cfg(cfg, item)) {
        option::some(item)
    } else {
        option::none
    }
}

fn fold_native_mod(&ast::crate_cfg cfg, &ast::native_mod nm,
                   fold::ast_fold fld) -> ast::native_mod {
    auto filter = bind filter_native_item(cfg, _);
    auto filtered_items = ivec::filter_map(filter, nm.items);
    ret rec(native_name=nm.native_name,
            abi=nm.abi,
            view_items=ivec::map(fld.fold_view_item, nm.view_items),
            items=filtered_items);
}

fn filter_stmt(&ast::crate_cfg cfg,
               &@ast::stmt stmt) -> option::t[@ast::stmt] {
    alt (stmt.node) {
        case (ast::stmt_decl(?decl, _)) {
            alt (decl.node) {
                case (ast::decl_item(?item)) {
                    if (item_in_cfg(cfg, item)) {
                        option::some(stmt)
                    } else {
                        option::none
                    }
                }
                case (_) { option::some(stmt) }
            }
        }
        case (_) { option::some(stmt) }
    }
}

fn fold_block(&ast::crate_cfg cfg, &ast::blk_ b,
              fold::ast_fold fld) -> ast::blk_  {
    auto filter = bind filter_stmt(cfg, _);
    auto filtered_stmts = ivec::filter_map(filter, b.stmts);
    ret rec(stmts=ivec::map(fld.fold_stmt, filtered_stmts),
            expr=option::map(fld.fold_expr, b.expr),
            id=b.id);
}

fn item_in_cfg(&ast::crate_cfg cfg, &@ast::item item) -> bool {
    ret in_cfg(cfg, item.attrs);
}

fn native_item_in_cfg(&ast::crate_cfg cfg, &@ast::native_item item) -> bool {
    ret in_cfg(cfg, item.attrs);
}

// Determine if an item should be translated in the current crate
// configuration based on the item's attributes
fn in_cfg(&ast::crate_cfg cfg, &ast::attribute[] attrs) -> bool {

    // The "cfg" attributes on the item
    auto item_cfg_attrs = attr::find_attrs_by_name(attrs, "cfg");
    auto item_has_cfg_attrs = ivec::len(item_cfg_attrs) > 0u;
    if (!item_has_cfg_attrs) { ret true; }

    // Pull the inner meta_items from the #[cfg(meta_item, ...)]  attributes,
    // so we can match against them. This is the list of configurations for
    // which the item is valid
    auto item_cfg_metas = {
        fn extract_metas(&(@ast::meta_item)[] inner_items,
                         &@ast::meta_item cfg_item)
        -> (@ast::meta_item)[] {

            alt (cfg_item.node) {
                case (ast::meta_list(?name, ?items)) {
                    assert name == "cfg";
                    inner_items + items
                }
                case (_) { inner_items }
            }
        }
        auto cfg_metas = attr::attr_metas(item_cfg_attrs);
        ivec::foldl(extract_metas, ~[], cfg_metas)
    };

    for (@ast::meta_item cfg_mi in item_cfg_metas) {
        if (attr::contains(cfg, cfg_mi)) { ret true; }
    }

    ret false;
}


// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
