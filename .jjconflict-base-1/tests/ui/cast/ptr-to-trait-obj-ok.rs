//@ check-pass

// Casting pointers to object types has some special rules in order to
// ensure VTables stay valid. E.g.
// - Cannot introduce new autotraits
// - Cannot extend or shrink lifetimes in trait arguments
// - Cannot extend the lifetime of the object type
//
// This test is a mostly miscellaneous set of examples of casts that do
// uphold these rules

trait Trait<'a> {}

fn remove_auto<'a>(x: *mut (dyn Trait<'a> + Send)) -> *mut dyn Trait<'a> {
    x as _
}

fn cast_inherent_lt<'a: 'b, 'b>(
    x: *mut (dyn Trait<'static> + 'a)
) -> *mut (dyn Trait<'static> + 'b) {
    x as _
}

fn cast_away_higher_ranked<'a>(x: *mut dyn for<'b> Trait<'b>) -> *mut dyn Trait<'a> {
    x as _
}

fn unprincipled<'a: 'b, 'b>(x: *mut (dyn Send + 'a)) -> *mut (dyn Sync + 'b) {
    x as _
}

fn remove_principal<'a: 'b, 'b, 't>(x: *mut (dyn Trait<'t> + Send + 'a)) -> *mut (dyn Send + 'b) {
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

fn cast_inherent_lt_wrap<'a: 'b, 'b>(
    x: *mut (dyn Trait<'static> + 'a),
) -> *mut Wrapper<dyn Trait<'static> + 'b> {
    x as _
}

fn cast_away_higher_ranked_wrap<'a>(x: *mut dyn for<'b> Trait<'b>) -> *mut Wrapper<dyn Trait<'a>> {
    x as _
}

fn unprincipled_wrap<'a: 'b, 'b>(x: *mut (dyn Send + 'a)) -> *mut Wrapper<dyn Sync + 'b> {
    x as _
}

fn main() {}
