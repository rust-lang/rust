use syntax::{ast, fold, attr};

export strip_unconfigured_items;
export metas_in_cfg;
export strip_items;

type in_cfg_pred = fn@(~[ast::attribute]) -> bool;

type ctxt = @{
    in_cfg: in_cfg_pred
};

// Support conditional compilation by transforming the AST, stripping out
// any items that do not belong in the current configuration
fn strip_unconfigured_items(crate: @ast::crate) -> @ast::crate {
    do strip_items(crate) |attrs| {
        in_cfg(crate.node.config, attrs)
    }
}

fn strip_items(crate: @ast::crate, in_cfg: in_cfg_pred)
    -> @ast::crate {

    let ctxt = @{in_cfg: in_cfg};

    let precursor =
        @{fold_mod: |a,b| fold_mod(ctxt, a, b),
          fold_block: fold::wrap(|a,b| fold_block(ctxt, a, b) ),
          fold_foreign_mod: |a,b| fold_foreign_mod(ctxt, a, b),
          .. *fold::default_ast_fold()};

    let fold = fold::make_fold(precursor);
    let res = @fold.fold_crate(*crate);
    return res;
}

fn filter_item(cx: ctxt, &&item: @ast::item) ->
   Option<@ast::item> {
    if item_in_cfg(cx, item) { option::Some(item) } else { option::None }
}

fn filter_view_item(cx: ctxt, &&view_item: @ast::view_item
                   )-> Option<@ast::view_item> {
    if view_item_in_cfg(cx, view_item) {
        option::Some(view_item)
    } else {
        option::None
    }
}

fn fold_mod(cx: ctxt, m: ast::_mod, fld: fold::ast_fold) ->
   ast::_mod {
    let item_filter = |a| filter_item(cx, a);
    let filtered_items = vec::filter_map(m.items, item_filter);
    let view_item_filter = |a| filter_view_item(cx, a);
    let filtered_view_items = vec::filter_map(m.view_items, view_item_filter);
    return {
        view_items: vec::map(filtered_view_items, |x| fld.fold_view_item(x)),
        items: vec::filter_map(filtered_items, |x| fld.fold_item(x))
    };
}

fn filter_foreign_item(cx: ctxt, &&item: @ast::foreign_item) ->
   Option<@ast::foreign_item> {
    if foreign_item_in_cfg(cx, item) {
        option::Some(item)
    } else { option::None }
}

fn fold_foreign_mod(cx: ctxt, nm: ast::foreign_mod,
                   fld: fold::ast_fold) -> ast::foreign_mod {
    let item_filter = |a| filter_foreign_item(cx, a);
    let filtered_items = vec::filter_map(nm.items, item_filter);
    let view_item_filter = |a| filter_view_item(cx, a);
    let filtered_view_items = vec::filter_map(
        nm.view_items, view_item_filter);
    return {
        sort: nm.sort,
        view_items: vec::map(filtered_view_items, |x| fld.fold_view_item(x)),
        items: filtered_items
    };
}

fn filter_stmt(cx: ctxt, &&stmt: @ast::stmt) ->
   Option<@ast::stmt> {
    match stmt.node {
      ast::stmt_decl(decl, _) => {
        match decl.node {
          ast::decl_item(item) => {
            if item_in_cfg(cx, item) {
                option::Some(stmt)
            } else { option::None }
          }
          _ => option::Some(stmt)
        }
      }
      _ => option::Some(stmt)
    }
}

fn fold_block(cx: ctxt, b: ast::blk_, fld: fold::ast_fold) ->
   ast::blk_ {
    let filter = |a| filter_stmt(cx, a);
    let filtered_stmts = vec::filter_map(b.stmts, filter);
    return {view_items: b.view_items,
         stmts: vec::map(filtered_stmts, |x| fld.fold_stmt(x)),
         expr: option::map(b.expr, |x| fld.fold_expr(x)),
         id: b.id,
         rules: b.rules};
}

fn item_in_cfg(cx: ctxt, item: @ast::item) -> bool {
    return cx.in_cfg(item.attrs);
}

fn foreign_item_in_cfg(cx: ctxt, item: @ast::foreign_item) -> bool {
    return cx.in_cfg(item.attrs);
}

fn view_item_in_cfg(cx: ctxt, item: @ast::view_item) -> bool {
    return cx.in_cfg(item.attrs);
}

// Determine if an item should be translated in the current crate
// configuration based on the item's attributes
fn in_cfg(cfg: ast::crate_cfg, attrs: ~[ast::attribute]) -> bool {
    metas_in_cfg(cfg, attr::attr_metas(attrs))
}

fn metas_in_cfg(cfg: ast::crate_cfg, metas: ~[@ast::meta_item]) -> bool {

    // The "cfg" attributes on the item
    let cfg_metas = attr::find_meta_items_by_name(metas, ~"cfg");

    // Pull the inner meta_items from the #[cfg(meta_item, ...)]  attributes,
    // so we can match against them. This is the list of configurations for
    // which the item is valid
    let cfg_metas = vec::concat(vec::filter_map(cfg_metas,
        |&&i| attr::get_meta_item_list(i) ));

    let has_cfg_metas = vec::len(cfg_metas) > 0u;
    if !has_cfg_metas { return true; }

    for cfg_metas.each |cfg_mi| {
        if attr::contains(cfg, *cfg_mi) { return true; }
    }

    return false;
}


// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
