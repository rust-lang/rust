//@ check-pass

trait Trait<'a> {}

fn remove_auto<'a>(x: *mut (dyn Trait<'a> + Send)) -> *mut dyn Trait<'a> {
    x as _
}

fn cast_inherent_lt<'a, 'b>(x: *mut (dyn Trait<'static> + 'a)) -> *mut (dyn Trait<'static> + 'b) {
    x as _
}

fn cast_away_higher_ranked<'a>(x: *mut dyn for<'b> Trait<'b>) -> *mut dyn Trait<'a> {
    x as _
}

fn unprincipled<'a, 'b>(x: *mut (dyn Send + 'a)) -> *mut (dyn Sync + 'b) {
    x as _
}

// If it is possible to coerce from the source to the target type modulo
// regions, then we skip the HIR checks for ptr-to-ptr casts and possibly
// insert an unsizing coercion into the MIR before the ptr-to-ptr cast.
// By wrapping the target type, we ensure that no coercion happens
// and also test the non-coercion cast behavior.
struct Wrapper<T: ?Sized>(T);

fn remove_auto_wrap<'a>(x: *mut (dyn Trait<'a> + Send)) -> *mut Wrapper<dyn Trait<'a>> {
    x as _
}

fn cast_inherent_lt_wrap<'a, 'b>(
    x: *mut (dyn Trait<'static> + 'a),
) -> *mut Wrapper<dyn Trait<'static> + 'b> {
    x as _
}

fn cast_away_higher_ranked_wrap<'a>(x: *mut dyn for<'b> Trait<'b>) -> *mut Wrapper<dyn Trait<'a>> {
    x as _
}

fn unprincipled_wrap<'a, 'b>(x: *mut (dyn Send + 'a)) -> *mut Wrapper<dyn Sync + 'b> {
    x as _
}

fn main() {}
