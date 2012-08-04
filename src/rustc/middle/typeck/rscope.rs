import result::result;

trait region_scope {
    fn anon_region() -> result<ty::region, ~str>;
    fn named_region(id: ast::ident) -> result<ty::region, ~str>;
}

enum empty_rscope { empty_rscope }
impl of region_scope for empty_rscope {
    fn anon_region() -> result<ty::region, ~str> {
        result::ok(ty::re_static)
    }
    fn named_region(id: ast::ident) -> result<ty::region, ~str> {
        if *id == ~"static" { result::ok(ty::re_static) }
        else { result::err(~"only the static region is allowed here") }
    }
}

enum type_rscope = bool;
impl of region_scope for type_rscope {
    fn anon_region() -> result<ty::region, ~str> {
        if *self {
            result::ok(ty::re_bound(ty::br_self))
        } else {
            result::err(~"to use region types here, the containing type \
                         must be declared with a region bound")
        }
    }
    fn named_region(id: ast::ident) -> result<ty::region, ~str> {
        do empty_rscope.named_region(id).chain_err |_e| {
            if *id == ~"self" { self.anon_region() }
            else {
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
impl of region_scope for @anon_rscope {
    fn anon_region() -> result<ty::region, ~str> {
        result::ok(self.anon)
    }
    fn named_region(id: ast::ident) -> result<ty::region, ~str> {
        self.base.named_region(id)
    }
}

enum binding_rscope = {base: region_scope};
fn in_binding_rscope<RS: region_scope copy owned>(self: RS)
    -> @binding_rscope {
    let base = self as region_scope;
    @binding_rscope({base: base})
}
impl of region_scope for @binding_rscope {
    fn anon_region() -> result<ty::region, ~str> {
        result::ok(ty::re_bound(ty::br_anon))
    }
    fn named_region(id: ast::ident) -> result<ty::region, ~str> {
        do self.base.named_region(id).chain_err |_e| {
            result::ok(ty::re_bound(ty::br_named(id)))
        }
    }
}
