import core::{vec, option};
import syntax::{ast, fold};
import attr;

export strip_unconfigured_items;
export metas_in_cfg;

// Support conditional compilation by transforming the AST, stripping out
// any items that do not belong in the current configuration
fn strip_unconfigured_items(crate: @ast::crate) -> @ast::crate {
    let cfg = crate.node.config;

    let precursor =
        {fold_mod: bind fold_mod(cfg, _, _),
         fold_block: bind fold_block(cfg, _, _),
         fold_native_mod: bind fold_native_mod(cfg, _, _)
            with *fold::default_ast_fold()};

    let fold = fold::make_fold(precursor);
    let res = @fold.fold_crate(*crate);
    ret res;
}

fn filter_item(cfg: ast::crate_cfg, &&item: @ast::item) ->
   option::t<@ast::item> {
    if item_in_cfg(cfg, item) { option::some(item) } else { option::none }
}

fn fold_mod(cfg: ast::crate_cfg, m: ast::_mod, fld: fold::ast_fold) ->
   ast::_mod {
    let filter = bind filter_item(cfg, _);
    let filtered_items = vec::filter_map(filter, m.items);
    ret {view_items: vec::map(fld.fold_view_item, m.view_items),
         items: vec::map(fld.fold_item, filtered_items)};
}

fn filter_native_item(cfg: ast::crate_cfg, &&item: @ast::native_item) ->
   option::t<@ast::native_item> {
    if native_item_in_cfg(cfg, item) {
        option::some(item)
    } else { option::none }
}

fn fold_native_mod(cfg: ast::crate_cfg, nm: ast::native_mod,
                   fld: fold::ast_fold) -> ast::native_mod {
    let filter = bind filter_native_item(cfg, _);
    let filtered_items = vec::filter_map(filter, nm.items);
    ret {view_items: vec::map(fld.fold_view_item, nm.view_items),
         items: filtered_items};
}

fn filter_stmt(cfg: ast::crate_cfg, &&stmt: @ast::stmt) ->
   option::t<@ast::stmt> {
    alt stmt.node {
      ast::stmt_decl(decl, _) {
        alt decl.node {
          ast::decl_item(item) {
            if item_in_cfg(cfg, item) {
                option::some(stmt)
            } else { option::none }
          }
          _ { option::some(stmt) }
        }
      }
      _ { option::some(stmt) }
    }
}

fn fold_block(cfg: ast::crate_cfg, b: ast::blk_, fld: fold::ast_fold) ->
   ast::blk_ {
    let filter = bind filter_stmt(cfg, _);
    let filtered_stmts = vec::filter_map(filter, b.stmts);
    ret {view_items: b.view_items,
         stmts: vec::map(fld.fold_stmt, filtered_stmts),
         expr: option::map(fld.fold_expr, b.expr),
         id: b.id,
         rules: b.rules};
}

fn item_in_cfg(cfg: ast::crate_cfg, item: @ast::item) -> bool {
    ret in_cfg(cfg, item.attrs);
}

fn native_item_in_cfg(cfg: ast::crate_cfg, item: @ast::native_item) -> bool {
    ret in_cfg(cfg, item.attrs);
}

// Determine if an item should be translated in the current crate
// configuration based on the item's attributes
fn in_cfg(cfg: ast::crate_cfg, attrs: [ast::attribute]) -> bool {
    metas_in_cfg(cfg, attr::attr_metas(attrs))
}

fn metas_in_cfg(cfg: ast::crate_cfg, metas: [@ast::meta_item]) -> bool {

    // The "cfg" attributes on the item
    let cfg_metas = attr::find_meta_items_by_name(metas, "cfg");

    // Pull the inner meta_items from the #[cfg(meta_item, ...)]  attributes,
    // so we can match against them. This is the list of configurations for
    // which the item is valid
    let cfg_metas = vec::concat(vec::filter_map(
        {|&&i| attr::get_meta_item_list(i)}, cfg_metas));

    let has_cfg_metas = vec::len(cfg_metas) > 0u;
    if !has_cfg_metas { ret true; }

    for cfg_mi: @ast::meta_item in cfg_metas {
        if attr::contains(cfg, cfg_mi) { ret true; }
    }

    ret false;
}


// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
