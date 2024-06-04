//@ check-pass

trait Trait<'a> {}

fn remove_auto<'a>(x: *mut (dyn Trait<'a> + Send)) -> *mut dyn Trait<'a> {
    x as _
}

fn cast_inherent_lt<'a, 'b>(x: *mut (dyn Trait<'static> + 'a)) -> *mut (dyn Trait<'static> + 'b) {
    x as _
}

fn unprincipled<'a, 'b>(x: *mut (dyn Send + 'a)) -> *mut (dyn Sync + 'b) {
    x as _
}

fn main() {}
