//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#240. Hidden types should
// equate *before* inference var fallback, otherwise we can get mismatched types.

#[derive(Clone, Copy)]
struct FileSystem;
impl FileSystem {
    fn build<T>(self, commands: T) -> Option<impl Sized> {
        match false {
            true => Some(commands),
            false => {
                drop(match self.build::<_>(commands) {
                    Some(x) => x,
                    None => return None,
                });
                panic!()
            },
        }
    }
}

fn build2<T, U>() -> impl Sized {
    if false {
        build2::<U, T>()
    } else {
        loop {}
    };
    1u32
}

fn build3<'a>() -> impl Sized + use<'a> {
    if false {
        build3()
    } else {
        loop {}
    };
    1u32
}

fn main() {}
