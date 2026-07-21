// Normalize aliases before checking trivial bounds
// Regression test for #154145 and #140309

trait Trait {
    type Assoc;
}

trait Impossible {}

struct Inner<T>(T);

fn foo<T>()
where
    T: Trait<Assoc = String>,
    <T as Trait>::Assoc: Copy, //~ ERROR the trait bound `String: Copy` is not satisfied
{
}

fn bar()
where
    Inner<fn(&())>: Impossible,
    //~^ ERROR the trait bound `Inner<for<'a> fn(&'a ())>: Impossible` is not satisfied
{
}

fn main() {}
