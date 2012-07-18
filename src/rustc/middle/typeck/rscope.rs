import result::result;
import syntax::parse::token::special_idents;

trait region_scope {
    fn anon_region(span: span) -> result<ty::region, ~str>;
    fn named_region(span: span, id: ast::ident) -> result<ty::region, ~str>;
}

enum empty_rscope { empty_rscope }
impl empty_rscope: region_scope {
    fn anon_region(_span: span) -> result<ty::region, ~str> {
        result::ok(ty::re_static)
    }
    fn named_region(_span: span, id: ast::ident) -> result<ty::region, ~str> {
        if id == special_idents::static { result::ok(ty::re_static) }
        else { result::err(~"only the static region is allowed here") }
    }
}

enum type_rscope = bool;
impl type_rscope: region_scope {
    fn anon_region(_span: span) -> result<ty::region, ~str> {
        if *self {
            result::ok(ty::re_bound(ty::br_self))
        } else {
            result::err(~"to use region types here, the containing type \
                         must be declared with a region bound")
        }
    }
    fn named_region(span: span, id: ast::ident) -> result<ty::region, ~str> {
        do empty_rscope.named_region(span, id).chain_err |_e| {
            if id == special_idents::self_ {
                self.anon_region(span)
            } else {
                result::err(~"named regions other than `self` are not \
                             allowed as part of a type declaration")
            }
        }
    }
}

enum anon_rscope = {anon: ty::region, base: region_scope};
fn in_anon_rscope<RS: region_scope copy owned>(self: RS, r: ty::region)
    -> @anon_rscope {
    @anon_rscope({anon: r, base: self as region_scope})
}
impl @anon_rscope: region_scope {
    fn anon_region(_span: span) -> result<ty::region, ~str> {
        result::ok(self.anon)
    }
    fn named_region(span: span, id: ast::ident) -> result<ty::region, ~str> {
        self.base.named_region(span, id)
    }
}

struct binding_rscope {
    base: region_scope;
    mut anon_bindings: uint;
}
fn in_binding_rscope<RS: region_scope copy owned>(self: RS)
    -> @binding_rscope {
    let base = self as region_scope;
    @binding_rscope { base: base, anon_bindings: 0 }
}
impl @binding_rscope: region_scope {
    fn anon_region(_span: span) -> result<ty::region, ~str> {
        let idx = self.anon_bindings;
        self.anon_bindings += 1;
        result::ok(ty::re_bound(ty::br_anon(idx)))
    }
    fn named_region(span: span, id: ast::ident) -> result<ty::region, ~str> {
        do self.base.named_region(span, id).chain_err |_e| {
            result::ok(ty::re_bound(ty::br_named(id)))
        }
    }
}
