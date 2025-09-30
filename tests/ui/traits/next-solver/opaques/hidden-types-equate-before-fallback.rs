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

fn main() {}
