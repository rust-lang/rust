// Various tests where we over type parameters with multiple lifetime
// bounds.

trait SomeTrait { fn get(&self) -> isize; }


fn make_object_good1<'a,'b,A:SomeTrait+'a+'b>(v: A) -> Box<dyn SomeTrait + 'a> {
    // A outlives 'a AND 'b...
    Box::new(v) as Box<dyn SomeTrait + 'a> // ...hence this type is safe.
}

fn make_object_good2<'a,'b,A:SomeTrait+'a+'b>(v: A) -> Box<dyn SomeTrait + 'b> {
    // A outlives 'a AND 'b...
    Box::new(v) as Box<dyn SomeTrait + 'b> // ...hence this type is safe.
}

fn make_object_bad<'a,'b,'c,A:SomeTrait+'a+'b>(v: A) -> Box<dyn SomeTrait + 'c> {
    // A outlives 'a AND 'b...but not 'c.
    Box::new(v) as Box<dyn SomeTrait + 'a>
    //~^ ERROR lifetime may not live long enough
}

fn main() {
}
