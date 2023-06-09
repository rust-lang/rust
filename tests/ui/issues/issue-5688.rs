// run-pass
/*
# Corrupted initialization in the static struct

...should print &[1, 2, 3] but instead prints something like
&[4492532864, 24]. It is pretty evident that the compiler messed up
with the representation of [isize; n] and [isize] somehow, or at least
failed to typecheck correctly.
*/

#[derive(Copy, Clone)]
struct X { vec: &'static [isize] }

static V: &'static [X] = &[X { vec: &[1, 2, 3] }];

pub fn main() {
    for &v in V {
        println!("{:?}", v.vec);
    }
}
