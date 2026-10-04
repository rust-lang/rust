// Test cases involving principal-less traits (dyn Send without a primary trait).

struct Wrapper<T: ?Sized>(T);

// Cast to same auto trait

fn unprincipled<'a, 'b>(x: *mut (dyn Send + 'a)) -> *mut (dyn Send + 'b) {
    x as _
    //~^ ERROR: lifetime may not live long enough
}

fn unprincipled_static<'a>(x: *mut (dyn Send + 'a)) -> *mut (dyn Send + 'static) {
    x as _
    //~^ ERROR: lifetime may not live long enough
}

fn unprincipled_wrap<'a, 'b>(x: *mut (dyn Send + 'a)) -> *mut Wrapper<dyn Send + 'b> {
    x as _
    //~^ ERROR: lifetime may not live long enough
}

fn unprincipled_wrap_static<'a>(x: *mut (dyn Send + 'a)) -> *mut Wrapper<dyn Send + 'static> {
    x as _
    //~^ ERROR: lifetime may not live long enough
}

// Cast to different auto trait

fn unprincipled2<'a, 'b>(x: *mut (dyn Send + 'a)) -> *mut (dyn Sync + 'b) {
    x as _
    //~^ ERROR: lifetime may not live long enough
}

fn unprincipled2_static<'a>(x: *mut (dyn Send + 'a)) -> *mut (dyn Sync + 'static) {
    x as _
    //~^ ERROR: lifetime may not live long enough
}

fn unprincipled_wrap2<'a, 'b>(x: *mut (dyn Send + 'a)) -> *mut Wrapper<dyn Sync + 'b> {
    x as _
    //~^ ERROR: lifetime may not live long enough
}

fn unprincipled_wrap2_static<'a>(x: *mut (dyn Send + 'a)) -> *mut Wrapper<dyn Sync + 'static> {
    x as _
    //~^ ERROR: lifetime may not live long enough
}

// Cast away principal trait
trait Trait {}

fn unprincipled3<'a, 'b>(x: *mut (dyn Trait + Send + 'a)) -> *mut (dyn Send + 'b) {
    x as _
    //~^ ERROR: lifetime may not live long enough
}

fn unprincipled3_static<'a>(x: *mut (dyn Trait + Send + 'a)) -> *mut (dyn Send + 'static) {
    x as _
    //~^ ERROR: lifetime may not live long enough
}

fn unprincipled_wrap3<'a, 'b>(x: *mut (dyn Trait + Send + 'a)) -> *mut Wrapper<dyn Send + 'b> {
    x as _
    //~^ ERROR: casting `*mut (dyn Trait + Send + 'a)` as `*mut Wrapper<(dyn Send + 'b)>` is invalid
}

fn unprincipled_wrap3_static<'a>(
    x: *mut (dyn Trait + Send + 'a)
) -> *mut Wrapper<dyn Send + 'static> {
    x as _
    //~^ ERROR: casting `*mut (dyn Trait + Send + 'a)` as `*mut Wrapper<(dyn Send + 'static)>` is invalid
}

fn main() {}
