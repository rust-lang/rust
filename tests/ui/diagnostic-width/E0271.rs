//@ revisions: ascii unicode
//@[ascii] compile-flags: --diagnostic-width=40 -Zwrite-long-types-to-disk=yes
//@[unicode] compile-flags: -Zunstable-options --error-format=human-unicode --diagnostic-width=40 -Zwrite-long-types-to-disk=yes
trait Future {
    type Error;
}

impl<T, E> Future for Result<T, E> {
    type Error = E;
}

impl<T> Future for Option<T> {
    type Error = ();
}

struct Foo;

fn foo() -> Box<dyn Future<Error=Foo>> {
    Box::new( //[ascii]~ ERROR E0271
        Ok::<_, ()>(
            Err::<(), _>(
                Ok::<_, ()>(
                    Err::<(), _>(
                        Ok::<_, ()>(
                            Err::<(), _>(Some(5))
                        )
                    )
                )
            )
        )
    )
}
fn main() {
}
