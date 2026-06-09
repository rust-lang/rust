// Checks whether borrowing is suggested when a trait bound is not satisfied
// for found type `T`, but is for `&/&mut T`.

fn main() {
    let f = Fancy{};
    let o = Other::new(f);
    //~^ ERROR: the trait bound `Fancy: SomeTrait` is not satisfied [E0277]
}

struct Fancy {}

impl <'a> SomeTrait for &'a Fancy {
}

trait SomeTrait {}

struct Other<'a, G> {
    a: &'a str,
    g: G,
}

// Broadly copied from https://docs.rs/petgraph/0.5.1/src/petgraph/dot.rs.html#70
impl<'a, G> Other<'a, G>
where
    G: SomeTrait,
{
    pub fn new(g: G) -> Self {
        Other {
            a: "hi",
            g: g,
        }
    }
}
