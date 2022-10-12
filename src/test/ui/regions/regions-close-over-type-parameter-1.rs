// Test for what happens when a type parameter `A` is closed over into
// an object. This should yield errors unless `A` (and the object)
// both have suitable bounds.

trait SomeTrait {
    fn get(&self) -> isize;
}


fn make_object1<A: SomeTrait>(v: A) -> Box<dyn SomeTrait + 'static> {
    Box::new(v) as Box<dyn SomeTrait + 'static>
    //~^ ERROR the parameter type `A` may not live long enough
}

fn make_object2<'a, A: SomeTrait + 'a>(v: A) -> Box<dyn SomeTrait + 'a> {
    Box::new(v) as Box<dyn SomeTrait + 'a>
}

fn make_object3<'a, 'b, A: SomeTrait + 'a>(v: A) -> Box<dyn SomeTrait + 'b> {
    Box::new(v) as Box<dyn SomeTrait + 'b>
    //~^ ERROR the parameter type `A` may not live long enough
}

fn main() {}
